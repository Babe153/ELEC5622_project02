"""Prepare an independent epoch extension. Default is a read-only space check.
Never trains, evaluates a model, modifies the source or uses shared/hard-linked checkpoints.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def plan_extension(source, destination, epochs):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be independent sibling/external directories')
    if destination.exists() or destination.with_name(destination.name + '.preparing').exists():
        raise FileExistsError('Destination or preparation staging directory already exists; refusing overwrite')
    if not (source/'completed.json').exists():
        raise ValueError('Source experiment must have completed')
    cfg = read_json(source/'config.json')
    manifest = read_json(source/'resume_format.json')
    if manifest.get('version') != 2:
        raise ValueError('Source must use full v2 resume checkpoints')
    if manifest['config'] != {k:v for k,v in cfg.items() if k!='epochs'}:
        raise ValueError('Source config differs from its resume manifest')
    if epochs <= cfg['epochs']:
        raise ValueError('Target epoch budget must exceed the source budget')
    trials = read_json(source/'trials.json')
    expected = {(f'{m}_{st}',f'lr_{lr:g}') for m in cfg['models'] for st in cfg['strategies'] for lr in cfg['learning_rates']}
    if len(trials)!=len(expected) or {(t['group'],t['trial_id']) for t in trials} != expected:
        raise ValueError('Source trial inventory is incomplete or duplicated')
    paths = {source/n for n in ['config.json','resume_format.json','classes.json','environment.json','weights.json','trials.json','trials.csv']}
    for name in ['loader_upgrade.json']:
        if (source/name).exists(): paths.add(source/name)
    for group in {t['group'] for t in trials}:
        paths.update([source/group/'best.pt',source/group/'selection.json'])
    for t in trials:
        if t['status'] not in ['completed','diverged']:
            raise ValueError('Source contains a nonterminal trial')
        if t['status']=='completed' and t['epochs_completed']!=cfg['epochs']:
            raise ValueError('Completed trial does not reach source epoch budget')
        paths.add(source/t['group']/(t['trial_id']+'_last.pt'))
        if t['epochs_completed']:
            paths.add(source/t['group']/(t['trial_id']+'_history.csv'))
        migration=source/t['group']/(t['trial_id']+'_loader_upgrade.json')
        if migration.exists(): paths.add(migration)
    for p in paths:
        if not p.is_file(): raise FileNotFoundError(p)
    total = sum(p.stat().st_size for p in paths)
    # Atomic checkpoint replacement + 512 MiB buffer + small history/metadata growth.
    temporary = max(p.stat().st_size for p in paths if p.name.endswith('_last.pt'))
    reserve = 512*1024**2 + 32*1024**2
    needed = total + temporary + reserve
    disk = destination.parent
    while not disk.exists(): disk=disk.parent
    free = shutil.disk_usage(disk).free
    return {'source':str(source),'destination':str(destination),'source_epochs':cfg['epochs'],
            'target_epochs':epochs,'copy_bytes':total,'atomic_save_bytes':temporary,'reserve_bytes':reserve,
            'required_free_bytes':needed,'available_bytes':free,'shortfall_bytes':max(0,needed-free),
            'continuing_trials':sum(t['status']=='completed' for t in trials),
            'skipped_diverged_trials':sum(t['status']=='diverged' for t in trials),
            'files':[p.relative_to(source).as_posix() for p in sorted(paths)]}


def prepare_extension(plan):
    # Recheck immediately before mutation; fail before making a partial destination.
    plan = plan_extension(plan['source'],plan['destination'],plan['target_epochs'])
    if plan['shortfall_bytes']:
        raise OSError(f"Insufficient space: free another {plan['shortfall_bytes']/1024**3:.2f} GiB")
    import torch
    source,destination=Path(plan['source']),Path(plan['destination'])
    trials=read_json(source/'trials.json')
    required={'model_state','optimizer_state','scaler_state','rng','sampler_generator',
              'train_generator','val_generator','history','elapsed_seconds','status','epoch'}
    validation=[]
    for t in trials:
        cp=torch.load(source/t['group']/(t['trial_id']+'_last.pt'),map_location='cpu',weights_only=True)
        if cp.get('version')!=2 or cp.get('identity')!=[t['group'],t['learning_rate']] or not required<=cp.keys():
            raise ValueError('Checkpoint format/identity is invalid')
        if cp['epoch']!=t['epochs_completed'] or len(cp['history'])!=cp['epoch'] or cp['status']!=t['status']:
            raise ValueError('Checkpoint epoch/status disagrees with trial summary')
        if [r['epoch'] for r in cp['history']]!=list(range(1,cp['epoch']+1)):
            raise ValueError('Checkpoint epoch history is not contiguous')
        validation.append({'group':t['group'],'trial_id':t['trial_id'],'saved_epoch':cp['epoch'],
                           'next_epoch':cp['epoch']+1 if cp['status']=='completed' else None,'status':cp['status']})
        del cp
    staging=destination.with_name(destination.name+'.preparing')
    staging.mkdir(parents=True,exist_ok=False)
    checks=[]
    for rel in plan['files']:
        src=source/rel; dest=staging/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        before=digest(src);shutil.copy2(src,dest)
        if digest(dest)!=before or digest(src)!=before:
            raise RuntimeError(f'Source changed or copy verification failed: {rel}; staging retained for inspection')
        checks.append({'path':rel,'source_sha256':before})
    # Preserve selected models/metrics as baseline evidence, not active test/completion markers.
    for name in ['selected_models.json','completed.json','test_results.json','test_results.csv']:
        if (source/name).exists():
            target=staging/'baseline_60_summary'/name;target.parent.mkdir(exist_ok=True)
            shutil.copy2(source/name,target)
            if digest(target)!=digest(source/name): raise RuntimeError('Baseline summary copy mismatch')
    cfg=read_json(staging/'config.json');cfg['epochs']=plan['target_epochs'];write_json(staging/'config.json',cfg)
    write_json(staging/'extension_provenance.json',{
        'created_utc':datetime.now(timezone.utc).isoformat(),'source_experiment':str(source),
        'source_epochs':plan['source_epochs'],'target_epochs':plan['target_epochs'],
        'source_test_previously_evaluated':(source/'test_results.json').exists(),
        'purpose':'Additional exploratory validation comparison; not an untouched-test confirmatory study',
        'copy_strategy':'Independent physical copies; no hard links or symlinks',
        'training_started_by_preparation':False,'checkpoints':validation,'source_file_hashes':checks})
    write_json(staging/'prepared.json',{'status':'prepared_not_started','target_epochs':plan['target_epochs'],
                                      'continuing_trials':plan['continuing_trials'],
                                      'skipped_diverged_trials':plan['skipped_diverged_trials']})
    # A second guard: an accidental resume exits at the saved boundary until explicitly enabled.
    (staging/'pause.request').write_text('Prepared only. Remove this file only when training is explicitly requested.\n')
    staging.rename(destination)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/baseline_60epochs')
    parser.add_argument('--destination',type=Path,default=ROOT/'results/baseline_100epochs')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--prepare',action='store_true',help='Copy and verify files; still never starts training')
    args=parser.parse_args()
    plan=plan_extension(args.source,args.destination,args.epochs)
    for key in ['copy_bytes','atomic_save_bytes','reserve_bytes','required_free_bytes','available_bytes','shortfall_bytes']:
        print(f'{key}: {plan[key]/1024**3:.3f} GiB')
    print(f"Continue {plan['continuing_trials']} trials; skip {plan['skipped_diverged_trials']} diverged trial(s).")
    if args.prepare:
        prepare_extension(plan); print(f'Prepared and paused: {args.destination}. No training started.')
    else: print('Read-only plan; no experiment files copied. Add --prepare to prepare without training.')

if __name__=='__main__':main()
