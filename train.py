"""HEp-2 assignment: local pretrained weights, validation-only model selection.

No action is performed on import. `check` never trains; `train` never evaluates
test images; `test` evaluates the four previously selected validation winners.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
WEIGHTS = {"alexnet": ("alexnet-owt-7be5be79.pth", "7be5be79"),
           "resnet34": ("resnet34-b627a593.pth", "b627a593")}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve(config, key):
    return (ROOT / config[key]).resolve()


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    assert cfg["epochs"] > 0 and cfg["batch_size"] > 0
    assert cfg["num_workers"] >= 0 and cfg["image_size"] >= 63
    assert all(x > 0 and math.isfinite(x) for x in cfg["learning_rates"])
    assert set(cfg["models"]) <= set(WEIGHTS)
    assert set(cfg["strategies"]) <= {"partial", "full"}
    return cfg


def load_index(cfg):
    with resolve(cfg, "labels_csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    classes = sorted({r["Image class"].strip() for r in rows})
    if len(classes) != 6:
        raise ValueError(f"Expected six classes, found {classes}")
    labels = {}
    for row in rows:
        image_id = int(row["Image ID"])
        if image_id in labels:
            raise ValueError(f"Duplicate CSV ID: {image_id}")
        labels[image_id] = classes.index(row["Image class"].strip())
    splits, seen = {}, set()
    for split in ("training", "validation", "test"):
        files = sorted((resolve(cfg, "data_dir") / split).glob("*.png"))
        if len(files) != cfg["expected_counts"][split]:
            raise ValueError(f"{split}: found {len(files)} files; check config and dataset path")
        ids = [int(p.stem) for p in files]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Duplicate image IDs within/across provided splits")
        if not set(ids) <= labels.keys():
            raise ValueError(f"Images without labels in {split}")
        seen.update(ids)
        splits[split] = files
    if seen != labels.keys():
        raise ValueError("CSV IDs do not exactly match provided image splits")
    return classes, labels, splits


class Cells(Dataset):
    def __init__(self, files, labels, cfg, training=False):
        self.files, self.labels = files, labels
        ops = [transforms.Resize((cfg["image_size"], cfg["image_size"]),
                                 interpolation=transforms.InterpolationMode.BILINEAR)]
        if training:
            # One random variant per original image per epoch; no 10/40x expansion.
            ops.append(transforms.RandomRotation(cfg["rotation_degrees"],
                       interpolation=transforms.InterpolationMode.BILINEAR, fill=0))
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as image:
            x = self.transform(image.convert("RGB"))
        return x, self.labels[int(path.stem)]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(files, labels, cfg, device, training=False):
    generator = torch.Generator().manual_seed(cfg["seed"])
    return DataLoader(Cells(files, labels, cfg, training), batch_size=cfg["batch_size"],
                      shuffle=training, num_workers=cfg["num_workers"],
                      pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                      generator=generator, persistent_workers=cfg["num_workers"] > 0)


def verify_weights(cfg):
    records = []
    for name in cfg["models"]:
        filename, prefix = WEIGHTS[name]
        path = resolve(cfg, "weights_dir") / filename
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
        if not digest.startswith(prefix):
            raise ValueError(f"Official weight checksum mismatch: {path}")
        records.append({"model": name, "file": filename, "sha256": digest})
    return records


def build_model(name, strategy, cfg, pretrained=True):
    model = getattr(models, name)(weights=None)
    if pretrained:
        # Load the original 1000-class state first, then replace the output layer.
        state = torch.load(resolve(cfg, "weights_dir") / WEIGHTS[name][0],
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    if name == "alexnet":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 6)
    else:
        model.fc = nn.Linear(model.fc.in_features, 6)
    if strategy == "partial":
        for p in model.parameters():
            p.requires_grad = False
        train_layers = [model.classifier] if name == "alexnet" else [model.layer4, model.fc]
        for layer in train_layers:
            for p in layer.parameters():
                p.requires_grad = True
    return model


def training_mode(model):
    model.train()
    # Frozen ResNet blocks must also keep BatchNorm running statistics frozen.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def epoch_pass(model, batches, device, amp, optimizer=None, scaler=None):
    training = optimizer is not None
    training_mode(model) if training else model.eval()
    sync(device)
    started = time.perf_counter()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    count = 0
    with torch.set_grad_enabled(training):
        for x, y in batches:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite loss; this trial is stopped, not silently selected")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += loss.detach() * y.numel()
            correct += (logits.detach().argmax(1) == y).sum()
            count += y.numel()
    sync(device)
    seconds = time.perf_counter() - started
    return {"loss": loss_sum.item() / count, "accuracy": correct.item() / count,
            "seconds": seconds, "samples": count}


def environment(device):
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}


def check(cfg, device):
    classes, labels, splits = load_index(cfg)
    hashes = verify_weights(cfg)
    # Decode all training/validation images only. Do not run any test prediction.
    for split in ("training", "validation"):
        for path in splits[split]:
            with Image.open(path) as image:
                image.verify()
    x, _ = next(iter(loader(splits["validation"][:2], labels,
                           {**cfg, "num_workers": 0}, device)))
    checks = []
    for name in cfg["models"]:
        for strategy in cfg["strategies"]:
            seed_all(cfg["seed"])
            model = build_model(name, strategy, cfg).to(device)
            for param_name, param in model.named_parameters():
                expected = strategy == "full" or param_name.startswith(
                    ("classifier.",) if name == "alexnet" else ("layer4.", "fc."))
                assert param.requires_grad == expected, param_name
            training_mode(model)
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
                    assert not module.training
            model.eval()
            with torch.inference_mode():
                output = model(x.to(device))
            assert output.shape == (2, 6) and torch.isfinite(output).all()
            record = {"model": name, "strategy": strategy,
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "total_parameters": sum(p.numel() for p in model.parameters()),
                      "forward_shape": list(output.shape), "status": "passed"}
            checks.append(record)
            print(json.dumps(record), flush=True)
            del model, output
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    report = {"status": "passed", "formal_training_started": False,
              "optimizer_steps": 0, "test_predictions": 0, "environment": environment(device),
              "classes": classes, "counts": {s: len(f) for s, f in splits.items()},
              "weights": hashes, "configurations": checks}
    write_json(ROOT / "check_report.json", report)
    print("CHECK PASSED. No optimizer steps or test predictions performed.", flush=True)


def train(cfg, device, experiment):
    classes, labels, splits = load_index(cfg)
    hashes = verify_weights(cfg)
    run = resolve(cfg, "results_dir") / experiment
    # Never silently overwrite an earlier experiment.
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", cfg)
    write_json(run / "environment.json", environment(device))
    write_json(run / "weights.json", hashes)
    write_json(run / "classes.json", classes)
    amp = cfg["mixed_precision"] and device.type == "cuda"
    champions, summaries = {}, []
    for name in cfg["models"]:
        for strategy in cfg["strategies"]:
            group = f"{name}_{strategy}"
            group_dir = run / group
            group_dir.mkdir()
            champion_key = (-1, float("-inf"))
            for lr in cfg["learning_rates"]:
                seed_all(cfg["seed"])
                model = build_model(name, strategy, cfg).to(device)
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                                            lr=lr, momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
                scaler = torch.amp.GradScaler("cuda", enabled=amp)
                train_loader = loader(splits["training"], labels, cfg, device, True)
                val_loader = loader(splits["validation"], labels, cfg, device)
                trial_id = f"lr_{lr:g}"
                history, best_key, best_epoch = [], (-1, float("-inf")), 0
                trial_start = time.perf_counter()
                status, reason = "completed", ""
                for epoch in range(1, cfg["epochs"] + 1):
                    try:
                        tr = epoch_pass(model, train_loader, device, amp, optimizer, scaler)
                        va = epoch_pass(model, val_loader, device, amp)
                    except FloatingPointError as error:
                        status, reason = "diverged", str(error)
                        break
                    row = {"epoch": epoch, "learning_rate": lr,
                           "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
                           "val_loss": va["loss"], "val_accuracy": va["accuracy"],
                           "train_seconds": tr["seconds"], "validation_seconds": va["seconds"]}
                    history.append(row)
                    write_csv(group_dir / f"{trial_id}_history.csv", history)
                    key = (va["accuracy"], -va["loss"])
                    if key > best_key:
                        best_key, best_epoch = key, epoch
                    if key > champion_key:
                        champion_key = key
                        metadata = {"model": name, "strategy": strategy, "learning_rate": lr,
                                    "best_epoch": epoch, "val_accuracy": va["accuracy"],
                                    "val_loss": va["loss"], "trainable_parameters": trainable,
                                    "trial_id": trial_id, "classes": classes,
                                    "training_seconds_to_best": sum(r["train_seconds"] for r in history)}
                        payload = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                   "metadata": metadata, "config": cfg}
                        target = group_dir / "best.pt"
                        temporary = target.with_suffix(".tmp")
                        torch.save(payload, temporary)
                        os.replace(temporary, target)
                        champions[group] = metadata
                        write_json(group_dir / "selection.json", metadata)
                        del payload
                    print(f"{group} lr={lr:g} epoch={epoch}/{cfg['epochs']} "
                          f"train={tr['accuracy']:.4f} val={va['accuracy']:.4f} "
                          f"time={tr['seconds'] + va['seconds']:.1f}s", flush=True)
                summary = {"group": group, "trial_id": trial_id, "learning_rate": lr,
                           "epochs_completed": len(history), "status": status, "reason": reason,
                           "best_epoch": best_epoch,
                           "best_val_accuracy": best_key[0] if history else None,
                           "best_val_loss": -best_key[1] if history else None,
                           "trainable_parameters": trainable,
                           "training_seconds": sum(r["train_seconds"] for r in history),
                           "validation_seconds": sum(r["validation_seconds"] for r in history),
                           "trial_wall_seconds": time.perf_counter() - trial_start}
                summaries.append(summary)
                write_csv(run / "trials.csv", summaries)
                write_json(run / "trials.json", summaries)
                del model, optimizer, scaler, train_loader, val_loader
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if group not in champions:
                raise RuntimeError(f"No valid checkpoint for {group}; inspect trials.json")
            chosen = champions[group]
            chosen_trial = next(r for r in summaries if r["group"] == group and r["trial_id"] == chosen["trial_id"])
            chosen["selected_trial_training_seconds"] = chosen_trial["training_seconds"]
            chosen["selected_trial_validation_seconds"] = chosen_trial["validation_seconds"]
            chosen["selected_trial_status"] = chosen_trial["status"]
            chosen["group_tuning_wall_seconds"] = sum(r["trial_wall_seconds"] for r in summaries if r["group"] == group)
            write_json(group_dir / "selection.json", chosen)
    write_json(run / "selected_models.json", champions)
    write_json(run / "completed.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
                                       "test_evaluated": False})
    print(f"Training finished. Validation winners saved in {run}. Test has NOT been run.", flush=True)


def test(run, device):
    run = Path(run).resolve()
    if not (run / "completed.json").exists():
        raise ValueError("Training experiment is incomplete")
    if (run / "test_results.json").exists():
        raise FileExistsError("Test results already exist; refusing accidental repeated test evaluation")
    cfg = read_config(run / "config.json")
    classes, labels, splits = load_index(cfg)
    if classes != json.loads((run / "classes.json").read_text(encoding="utf-8")):
        raise ValueError("Class mapping changed")
    selected = json.loads((run / "selected_models.json").read_text(encoding="utf-8"))
    batches = loader(splits["test"], labels, cfg, device)
    results = []
    for group, selection in selected.items():
        checkpoint = torch.load(run / group / "best.pt", map_location="cpu", weights_only=True)
        if checkpoint["metadata"]["classes"] != classes:
            raise ValueError("Checkpoint class mapping mismatch")
        model = build_model(selection["model"], selection["strategy"], cfg, pretrained=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        del checkpoint
        model = model.to(device).eval()
        confusion = torch.zeros((6, 6), dtype=torch.long)
        predictions, offset = [], 0
        sync(device)
        start = time.perf_counter()
        with torch.inference_mode():
            for x, y in batches:
                with torch.autocast(device_type=device.type, enabled=cfg["mixed_precision"] and device.type == "cuda"):
                    pred = model(x.to(device)).argmax(1).cpu()
                confusion += torch.bincount(y * 6 + pred, minlength=36).reshape(6, 6)
                for truth, guess in zip(y.tolist(), pred.tolist()):
                    predictions.append({"image": splits["test"][offset].name,
                                        "true_class": classes[truth], "predicted_class": classes[guess]})
                    offset += 1
        sync(device)
        row = {"configuration": group, "test_accuracy": confusion.diag().sum().item() / confusion.sum().item(),
               "learning_rate": selection["learning_rate"], "best_epoch": selection["best_epoch"],
               "trainable_parameters": selection["trainable_parameters"],
               "training_seconds_to_best": selection["training_seconds_to_best"],
               "selected_trial_training_seconds": selection["selected_trial_training_seconds"],
               "group_tuning_wall_seconds": selection["group_tuning_wall_seconds"],
               "test_seconds": time.perf_counter() - start}
        results.append(row)
        write_csv(run / group / "test_predictions.csv", predictions)
        write_csv(run / group / "confusion_matrix.csv", [
            {"true_class": classes[i], **{classes[j]: confusion[i, j].item() for j in range(6)}} for i in range(6)])
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(run / "test_results.json", results)
    write_csv(run / "test_results.csv", results)
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "train", "test"])
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--experiment", default="baseline_30epochs")
    parser.add_argument("--run-dir", type=Path, help="Existing experiment directory for test")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Use the installed GPU environment; CPU use must be explicit.")
    if Path(args.experiment).name != args.experiment or args.experiment in {".", ".."}:
        parser.error("--experiment must be a simple folder name")
    device = torch.device(args.device)
    if args.command == "test":
        if args.run_dir is None:
            parser.error("test requires --run-dir")
        test(args.run_dir, device)
    else:
        cfg = read_config(args.config)
        seed_all(cfg["seed"])
        if args.command == "check":
            check(cfg, device)
        else:
            train(cfg, device, args.experiment)


if __name__ == "__main__":
    main()
