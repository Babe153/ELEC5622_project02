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
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
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
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)



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
        self.training, self.seed = training, cfg["seed"]
        self.degrees = cfg["rotation_degrees"]
        self.resize = transforms.Resize((cfg["image_size"], cfg["image_size"]),
                                       interpolation=transforms.InterpolationMode.BILINEAR)
        self.finish = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        # Epoch travels with each index, so prefetch never sees a stale shared epoch.
        epoch, index = key if isinstance(key, tuple) else (0, key)
        path = self.files[index]
        with Image.open(path) as image:
            image = self.resize(image.convert("RGB"))
            if self.training:
                # Local RNG: independent of worker lifetime, scheduling and global dropout RNG.
                token = f"{self.seed}:{epoch}:{path.stem}".encode("utf-8")
                sample_seed = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")
                angle = random.Random(sample_seed).uniform(-self.degrees, self.degrees)
                image = transforms.functional.rotate(image, angle,
                    interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
            x = self.finish(image)
        return x, self.labels[int(path.stem)]


class EpochSampler(Sampler):
    """Separate shuffle RNG from worker seed RNG; checkpoint its state each epoch."""
    def __init__(self, size, seed):
        self.size, self.epoch = size, 1
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.size

    def __iter__(self):
        order = torch.randperm(self.size, generator=self.generator).tolist()
        return iter((self.epoch, index) for index in order)


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
    sampler = EpochSampler(len(files), cfg["seed"]) if training else None
    return DataLoader(Cells(files, labels, cfg, training), batch_size=cfg["batch_size"],
                      sampler=sampler, num_workers=cfg["num_workers"],
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
    write_json(resolve(cfg, "results_dir") / "check_report.json", report)
    print("CHECK PASSED. No optimizer steps or test predictions performed.", flush=True)


def atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rng_state():
    # Plain Python values/tensors keep checkpoints compatible with weights_only=True.
    ns = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def check_space(run, cfg):
    # FP32 model + SGD momentum for trainable weights, all trial checkpoints,
    # four best models, one atomic replacement, and a 512 MiB reserve.
    counts = {("alexnet", "partial"): (57028422, 54558726),
              ("alexnet", "full"): (57028422, 57028422),
              ("resnet34", "partial"): (21287750, 13117446),
              ("resnet34", "full"): (21287750, 21287750)}
    sizes = [4 * sum(counts[(m, st)]) for m in cfg["models"] for st in cfg["strategies"]]
    total = sum(sizes) * len(cfg["learning_rates"])
    total += sum(4 * counts[(m, st)][0] for m in cfg["models"] for st in cfg["strategies"])
    total += max(sizes) + 512 * 1024**2
    existing = sum(p.stat().st_size for m in cfg["models"] for st in cfg["strategies"]
                   for p in (run / f"{m}_{st}").glob("*.pt")) if run.exists() else 0
    parent = run
    while not parent.exists():
        parent = parent.parent
    needed = max(0, total - existing)
    free = shutil.disk_usage(parent).free
    if free < needed:
        raise OSError(f"Insufficient checkpoint space: need about {needed / 1024**3:.2f} GiB free, "
                      f"have {free / 1024**3:.2f} GiB. Free space or change results_dir; no training started.")


def train(cfg, device, experiment=None, resume_run=None, target_epochs=None, upgrade_loader=False):
    resuming = resume_run is not None
    run = Path(resume_run).resolve() if resuming else resolve(cfg, "results_dir") / experiment
    if resuming:
        cfg = read_config(run / "config.json")
        if not (run / "resume_format.json").exists():
            raise ValueError("This legacy experiment has no complete resume checkpoints; start a new experiment.")
        manifest = json.loads((run / "resume_format.json").read_text(encoding="utf-8"))
        if manifest.get("version") not in (1, 2):
            raise ValueError("Unsupported resume format")
        if manifest["version"] == 1 and not upgrade_loader:
            raise ValueError("Loader v1 requires explicit --upgrade-loader; future rotation/shuffle sequence will change")
        if manifest.get("config") != {k: v for k, v in cfg.items() if k != "epochs"}:
            raise ValueError("Saved experiment settings changed; resume only allows a larger epoch budget")
        if (run / "test_results.json").exists():
            raise ValueError("Cannot resume an experiment after test evaluation.")
        if target_epochs is not None:
            if target_epochs < cfg["epochs"]:
                raise ValueError("--epochs cannot decrease the saved training budget")
            cfg = {**cfg, "epochs": target_epochs}
    check_space(run, cfg)
    classes, labels, splits = load_index(cfg)
    hashes = verify_weights(cfg)
    current_env = environment(device)
    if resuming:
        saved_env = json.loads((run / "environment.json").read_text(encoding="utf-8"))
        if saved_env != current_env:
            raise ValueError("Resume requires the same Python/PyTorch/CUDA/device environment")
        if hashes != json.loads((run / "weights.json").read_text(encoding="utf-8")):
            raise ValueError("Pretrained weights changed")
        if classes != json.loads((run / "classes.json").read_text(encoding="utf-8")):
            raise ValueError("Class mapping changed")
        if manifest["version"] == 1:
            write_json(run / "loader_upgrade.json", {
                "upgraded_utc": datetime.now(timezone.utc).isoformat(),
                "from": 1, "to": 2, "reason": "Persistent workers with deterministic per-image augmentation",
                "preserved": "model, optimizer, scaler, global RNG, history; shuffle RNG seeded from saved loader RNG",
                "random_sequence_changes": True})
            manifest.update(version=2, persistent_workers=True, allow_v1_checkpoint_migration=True,
                            augmentation="seed_epoch_image_blake2b", sampler="separate_epoch_sampler")
            write_json(run / "resume_format.json", manifest)
    else:
        manifest = {"allow_v1_checkpoint_migration": False}
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "environment.json", current_env)
        write_json(run / "weights.json", hashes)
        write_json(run / "classes.json", classes)
        write_json(run / "resume_format.json", {"version": 2, "boundary": "completed_epoch", "persistent_workers": True,
                                                      "augmentation": "seed_epoch_image_blake2b", "sampler": "separate_epoch_sampler",
                                                      "config": {k: v for k, v in cfg.items() if k != "epochs"}})
    (run / "paused.json").unlink(missing_ok=True)
    write_json(run / "config.json", cfg)
    (run / "completed.json").unlink(missing_ok=True)
    (run / "selected_models.json").unlink(missing_ok=True)
    amp = cfg["mixed_precision"] and device.type == "cuda"
    champions, summaries = {}, []
    for name in cfg["models"]:
        for strategy in cfg["strategies"]:
            group = f"{name}_{strategy}"
            group_dir = run / group
            group_dir.mkdir(exist_ok=True)
            champion_key = (-1, float("-inf"))
            if (group_dir / "best.pt").exists():
                previous = torch.load(group_dir / "best.pt", map_location="cpu", weights_only=True)
                champions[group] = previous["metadata"]
                champion_key = (champions[group]["val_accuracy"], -champions[group]["val_loss"])
                del previous
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
                checkpoint_path = group_dir / f"{trial_id}_last.pt"
                history, elapsed, status, reason = [], 0.0, "completed", ""
                if checkpoint_path.exists():
                    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    if cp["identity"] != [group, lr] or cp["version"] not in (1, 2):
                        raise ValueError("Checkpoint identity/version mismatch")
                    if cp["version"] == 1 and not manifest.get("allow_v1_checkpoint_migration", False):
                        raise ValueError("Legacy checkpoint requires explicit loader migration")
                    model.load_state_dict(cp["model_state"])
                    optimizer.load_state_dict(cp["optimizer_state"])
                    scaler.load_state_dict(cp["scaler_state"])
                    train_loader.generator.set_state(cp["train_generator"])
                    val_loader.generator.set_state(cp["val_generator"])
                    train_loader.sampler.generator.set_state(cp.get("sampler_generator", cp["train_generator"]))
                    history, elapsed = cp["history"], cp["elapsed_seconds"]
                    if cp["version"] == 1:
                        write_json(group_dir / f"{trial_id}_loader_upgrade.json", {
                            "last_legacy_epoch": len(history), "first_new_epoch": len(history) + 1,
                            "weights_optimizer_scaler_preserved": True})
                    print(f"Resuming {group} lr={lr:g}: {len(history)} epochs saved; target {cfg['epochs']}.", flush=True)
                    status, reason = cp["status"], cp["reason"]
                    restore_rng(cp["rng"])
                    del cp
                # CSV may be ahead after an interrupted write; checkpoint is authoritative.
                if history:
                    write_csv(group_dir / f"{trial_id}_history.csv", history)
                trial_start = time.perf_counter()

                def save_last():
                    atomic_save(checkpoint_path, {
                        "version": 2, "identity": [group, lr], "epoch": len(history),
                        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(), "rng": rng_state(),
                        "sampler_generator": train_loader.sampler.generator.get_state(),
                        "train_generator": train_loader.generator.get_state(),
                        "val_generator": val_loader.generator.get_state(),
                        "history": history, "status": status, "reason": reason,
                        "elapsed_seconds": elapsed + time.perf_counter() - trial_start})

                def promote_last():
                    nonlocal champion_key
                    if not history:
                        return
                    row = history[-1]
                    key = (row["val_accuracy"], -row["val_loss"])
                    if key > champion_key:
                        metadata = {"model": name, "strategy": strategy, "learning_rate": lr,
                                    "best_epoch": row["epoch"], "val_accuracy": row["val_accuracy"],
                                    "val_loss": row["val_loss"], "trainable_parameters": trainable,
                                    "trial_id": trial_id, "classes": classes,
                                    "training_seconds_to_best": sum(r["train_seconds"] for r in history)}
                        atomic_save(group_dir / "best.pt", {
                            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "metadata": metadata, "config": cfg})
                        champions[group], champion_key = metadata, key
                        write_json(group_dir / "selection.json", metadata)

                if status != "diverged":
                    # Repairs an interruption between saving last.pt and best.pt.
                    promote_last()
                    for epoch in range(len(history) + 1, cfg["epochs"] + 1):
                        if (run / "pause.request").exists():
                            write_json(run / "paused.json", {"group": group, "trial_id": trial_id,
                                                             "saved_epoch": len(history)})
                            print("Paused at a saved epoch boundary. Remove pause.request before resume.", flush=True)
                            return
                        train_loader.sampler.epoch = epoch
                        try:
                            tr = epoch_pass(model, train_loader, device, amp, optimizer, scaler)
                            va = epoch_pass(model, val_loader, device, amp)
                        except FloatingPointError as error:
                            status, reason = "diverged", str(error)
                            # Preserve the last valid boundary; terminal marker prevents retries.
                            if checkpoint_path.exists():
                                terminal = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                                terminal.update(status=status, reason=reason,
                                                elapsed_seconds=elapsed + time.perf_counter() - trial_start)
                                atomic_save(checkpoint_path, terminal)
                                del terminal
                            else:
                                save_last()
                            break
                        history.append({"epoch": epoch, "learning_rate": lr,
                                        "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
                                        "val_loss": va["loss"], "val_accuracy": va["accuracy"],
                                        "train_seconds": tr["seconds"], "validation_seconds": va["seconds"]})
                        save_last()
                        write_csv(group_dir / f"{trial_id}_history.csv", history)
                        promote_last()
                        print(f"{group} lr={lr:g} epoch={epoch}/{cfg['epochs']} "
                              f"train={tr['accuracy']:.4f} val={va['accuracy']:.4f} "
                              f"time={tr['seconds'] + va['seconds']:.1f}s", flush=True)
                best = max(history, key=lambda r: (r["val_accuracy"], -r["val_loss"])) if history else None
                summary = {"group": group, "trial_id": trial_id, "learning_rate": lr,
                           "epochs_completed": len(history), "status": status, "reason": reason,
                           "best_epoch": best["epoch"] if best else 0,
                           "best_val_accuracy": best["val_accuracy"] if best else None,
                           "best_val_loss": best["val_loss"] if best else None,
                           "trainable_parameters": trainable,
                           "training_seconds": sum(r["train_seconds"] for r in history),
                           "validation_seconds": sum(r["validation_seconds"] for r in history),
                           "trial_wall_seconds": elapsed + time.perf_counter() - trial_start}
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
    parser.add_argument("command", choices=["check", "train", "resume", "test"])
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--experiment", default="baseline_60epochs")
    parser.add_argument("--run-dir", type=Path, help="Existing experiment directory for resume/test")
    parser.add_argument("--upgrade-loader", action="store_true", help="Explicitly migrate v1 checkpoints to persistent loader v2")
    parser.add_argument("--epochs", type=int, help="Total target epochs for resume (not additional epochs)")
    args = parser.parse_args()
    if args.upgrade_loader and args.command != "resume":
        parser.error("--upgrade-loader is only valid with resume")
    if args.epochs is not None and args.command != "resume":
        parser.error("--epochs is only valid with resume; new training uses config.json")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Use the installed GPU environment; CPU use must be explicit.")
    if Path(args.experiment).name != args.experiment or args.experiment in {".", ".."}:
        parser.error("--experiment must be a simple folder name")
    device = torch.device(args.device)
    if args.command == "resume":
        if args.run_dir is None:
            parser.error("resume requires --run-dir")
        train(None, device, resume_run=args.run_dir, target_epochs=args.epochs, upgrade_loader=args.upgrade_loader)
    elif args.command == "test":
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
