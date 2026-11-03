"""Preparation tests use tiny fabricated checkpoints; never train or predict."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
spec=importlib.util.spec_from_file_location('extension',Path(__file__).resolve().parents[1]/'scripts/prepare_extension.py')
app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)

class PreparationTests(unittest.TestCase):
    def source(self, base):
        src=base/'source';src.mkdir()
        cfg={'epochs':2,'models':['alexnet'],'strategies':['full'],'learning_rates':[.001]}
        app.write_json(src/'config.json',cfg)
        app.write_json(src/'resume_format.json',{'version':2,'config':{k:v for k,v in cfg.items() if k!='epochs'}})
        for f in ['classes.json','environment.json','weights.json','completed.json','test_results.json','selected_models.json']:
            app.write_json(src/f,{})
        app.write_json(src/'trials.json',[{'group':'alexnet_full','trial_id':'lr_0.001','learning_rate':.001,'status':'completed','epochs_completed':2}])
        (src/'trials.csv').write_text('synthetic')
        gd=src/'alexnet_full';gd.mkdir()
        app.write_json(gd/'selection.json',{})
        (gd/'lr_0.001_history.csv').write_text('epoch\n1\n2\n')
        torch.save({'model_state':{'weight':torch.ones(1)}},gd/'best.pt')
        cp={'version':2,'identity':['alexnet_full',.001],'epoch':2,'status':'completed','history':[{'epoch':1},{'epoch':2}],
            'model_state':{'weight':torch.ones(1)},'optimizer_state':{},'scaler_state':{},'rng':{},
            'sampler_generator':torch.Generator().get_state(),'train_generator':torch.Generator().get_state(),
            'val_generator':torch.Generator().get_state(),'elapsed_seconds':0}
        torch.save(cp,gd/'lr_0.001_last.pt')
        return src

    def test_independent_copy_and_no_training(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);src=self.source(base);dest=base/'extension'
            before={p.relative_to(src).as_posix():app.digest(p) for p in src.rglob('*') if p.is_file()}
            plan=app.plan_extension(src,dest,4)
            self.assertFalse(dest.exists())
            app.prepare_extension(plan)
            self.assertEqual(app.read_json(dest/'config.json')['epochs'],4)
            self.assertEqual(app.read_json(src/'config.json')['epochs'],2)
            self.assertFalse((dest/'completed.json').exists())
            self.assertFalse((dest/'test_results.json').exists())
            self.assertTrue((dest/'baseline_60_summary/test_results.json').exists())
            self.assertTrue((dest/'pause.request').exists())
            self.assertFalse(app.read_json(dest/'extension_provenance.json')['training_started_by_preparation'])
            self.assertEqual(before,{p.relative_to(src).as_posix():app.digest(p) for p in src.rglob('*') if p.is_file()})
            cp='alexnet_full/lr_0.001_last.pt'
            self.assertFalse(os.path.samefile(src/cp,dest/cp))
            (dest/cp).write_bytes(b'changed copy')
            self.assertEqual(app.digest(src/cp),before[cp])
            with self.assertRaises(FileExistsError):app.plan_extension(src,dest,4)
            with self.assertRaises(ValueError):app.plan_extension(src,src/'nested',4)

    def test_shortage_fails_before_copy(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);src=self.source(base);dest=base/'extension'
            with patch.object(app.shutil,'disk_usage',return_value=type('Usage',(),{'free':0})()):
                plan=app.plan_extension(src,dest,4)
                self.assertGreater(plan['shortfall_bytes'],0)
                with self.assertRaises(OSError):app.prepare_extension(plan)
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name(dest.name+'.preparing').exists())

if __name__=='__main__':unittest.main()
