"""Synthetic CPU tests only: no assignment data, pretrained CNN or test-set access."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as app


def tiny_model(*args, **kwargs):
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, 12), nn.ReLU(), nn.Dropout(.4), nn.Linear(12, 6))


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


class ResumeTests(unittest.TestCase):
    def test_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            files = []
            for i in range(8):
                p = base / f'{i}.png'
                arr = np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(arr).save(p)
                files.append(p)
            index = (list('abcdef'), {i: i % 6 for i in range(8)},
                     {'training': files, 'validation': files[:4], 'test': []})
            cfg = {'results_dir': str(base), 'models':['alexnet'], 'strategies':['full'],
                   'learning_rates':[.001], 'epochs':3, 'batch_size':4, 'num_workers':2,
                   'seed':42, 'momentum':.9, 'weight_decay':.0005, 'mixed_precision':False,
                   'image_size':64, 'rotation_degrees':180}
            device = torch.device('cpu')
            with patch.object(app, 'build_model', tiny_model), patch.object(app, 'load_index', return_value=index), patch.object(app, 'verify_weights', return_value=[]), patch.object(app, 'check_space'):
                app.train(cfg, device, 'continuous')
                app.train({**cfg, 'epochs':2}, device, 'extended')
                app.train(None, device, resume_run=base/'extended', target_epochs=3)
                original_pass = app.epoch_pass
                calls = 0
                def interrupt(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 4:
                        raise RuntimeError('simulated interruption during second epoch')
                    return original_pass(*args, **kwargs)
                with patch.object(app, 'epoch_pass', side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, 'simulated'):
                        app.train(cfg, device, 'interrupted')
                self.assertFalse((base/'interrupted/completed.json').exists())
                app.train(None, device, resume_run=base/'interrupted')
                expected=load(base/'continuous/alexnet_full/lr_0.001_last.pt')
                for name in ['extended', 'interrupted']:
                    actual=load(base/name/'alexnet_full/lr_0.001_last.pt')
                    for k,v in expected['model_state'].items():
                        self.assertTrue(torch.equal(v, actual['model_state'][k]), k)
                    for k,v in expected['optimizer_state']['state'].items():
                        self.assertTrue(torch.equal(v['momentum_buffer'], actual['optimizer_state']['state'][k]['momentum_buffer']))
                    for a,b in zip(expected['history'], actual['history']):
                        for metric in ['train_loss','val_loss','train_accuracy','val_accuracy']:
                            self.assertEqual(a[metric], b[metric], (name,metric))
                    self.assertEqual(actual['epoch'],3)
                    self.assertTrue((base/name/'completed.json').exists())
                # A finished trial must not perform more optimizer steps on same-budget resume.
                with patch.object(app, 'epoch_pass', side_effect=AssertionError('unexpected epoch')):
                    app.train(None,device,resume_run=base/'extended')
                with self.assertRaises(ValueError):
                    app.train(None,device,resume_run=base/'extended',target_epochs=2)
                app.write_json(base/'extended/test_results.json', [])
                with self.assertRaisesRegex(ValueError, 'test evaluation'):
                    app.train(None,device,resume_run=base/'extended',target_epochs=4)
                # Legacy results cannot be used as full checkpoints.
                legacy=base/'legacy'; legacy.mkdir(); app.write_json(legacy/'config.json',cfg)
                with self.assertRaisesRegex(ValueError, 'legacy'):
                    app.train(None,device,resume_run=legacy)
                # Diverged trials are terminal, and must not be retried on resume.
                cp_path=base/'interrupted/alexnet_full/lr_0.001_last.pt'
                cp=load(cp_path); cp.update(status='diverged',reason='synthetic test')
                app.atomic_save(cp_path,cp)
                with patch.object(app,'epoch_pass',side_effect=AssertionError('retried divergence')):
                    app.train(None,device,resume_run=base/'interrupted',target_epochs=4)

                # Migrate a v1-shaped checkpoint without resetting weights or momentum.
                migration = base / 'migration'
                shutil.copytree(base/'continuous', migration)
                manifest=json.loads((migration/'resume_format.json').read_text())
                manifest.update(version=1, persistent_workers=False)
                app.write_json(migration/'resume_format.json',manifest)
                migrated_path=migration/'alexnet_full/lr_0.001_last.pt'
                old=load(migrated_path); old['version']=1; old.pop('sampler_generator')
                app.atomic_save(migrated_path,old)
                with self.assertRaisesRegex(ValueError,'upgrade-loader'):
                    app.train(None,device,resume_run=migration,target_epochs=4)
                def check_preserved(model,batches,device,amp,optimizer=None,scaler=None):
                    if optimizer is not None:
                        for k,v in old['model_state'].items():
                            self.assertTrue(torch.equal(v,model.state_dict()[k]))
                        for k,v in old['optimizer_state']['state'].items():
                            self.assertTrue(torch.equal(v['momentum_buffer'],optimizer.state_dict()['state'][k]['momentum_buffer']))
                    return original_pass(model,batches,device,amp,optimizer,scaler)
                with patch.object(app,'epoch_pass',side_effect=check_preserved):
                    app.train(None,device,resume_run=migration,target_epochs=4,upgrade_loader=True)
                converted=load(migrated_path)
                self.assertEqual(converted['version'],2)
                self.assertEqual(converted['history'][:3],old['history'])
                self.assertEqual(converted['epoch'],4)
                # Graceful pause leaves the last checkpoint intact and no completed marker.
                (migration/'pause.request').touch()
                with patch.object(app,'epoch_pass',side_effect=AssertionError('pause ignored')):
                    app.train(None,device,resume_run=migration,target_epochs=5)
                self.assertEqual(load(migrated_path)['epoch'],4)
                self.assertTrue((migration/'paused.json').exists())
                self.assertFalse((migration/'completed.json').exists())

    def test_space_check_is_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'new'
            with patch.object(app.shutil,'disk_usage',return_value=type('Usage',(),{'free':0})()):
                with self.assertRaisesRegex(OSError,'Insufficient checkpoint space'):
                    app.check_space(target, {'models':['alexnet'],'strategies':['full'],'learning_rates':[.001]})
            self.assertFalse(target.exists())

if __name__ == '__main__':
    unittest.main()
