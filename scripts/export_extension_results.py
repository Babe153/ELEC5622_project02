"""Publish validation-only extension evidence; never train or evaluate test data."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def jread(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def csvwrite(p,records):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
def jwrite(p,data):p.write_bytes((json.dumps(data,indent=2,ensure_ascii=False)+'\n').encode('utf-8'))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--baseline-dir',type=Path,default=ROOT/'results/baseline_60epochs')
    args=ap.parse_args();run=args.run_dir;old=args.baseline_dir;report=ROOT/'report'
    assert (run/'completed.json').exists()
    assert not (run/'test_results.json').exists(), 'This exporter describes validation-only extension evidence'
    assert jread(run/'config.json')['epochs']==100 and jread(old/'config.json')['epochs']==60
    preserved={p:sha(p) for p in (report/'experiments/baseline_60epochs').rglob('*') if p.is_file()}
    a=jread(old/'selected_models.json');b=jread(run/'selected_models.json')
    trials=jread(run/'trials.json');oldtrials={(t['group'],t['trial_id']):t for t in jread(old/'trials.json')}
    detailed=[]
    for t in trials:
        key=(t['group'],t['trial_id']);before=oldtrials[key]
        p=run/t['group']/(t['trial_id']+'_history.csv');rr=rows(p);prior=rows(old/t['group']/p.name)
        assert rr[:len(prior)]==prior,key
        extra=[r for r in rr if int(r['epoch'])>60]
        detailed.append({'configuration':t['group'],'learning_rate':t['learning_rate'],'status':t['status'],
                         'epochs_completed':t['epochs_completed'],'best_val_60':before['best_val_accuracy'],
                         'best_epoch_60':before['best_epoch'],'best_val_100':t['best_val_accuracy'],
                         'best_epoch_100':t['best_epoch'],'gain_percentage_points':100*(t['best_val_accuracy']-before['best_val_accuracy']),
                         'added_epochs':len(extra),'added_training_seconds':sum(float(r['train_seconds']) for r in extra),
                         'added_validation_seconds':sum(float(r['validation_seconds']) for r in extra),
                         'added_recorded_trial_wall_seconds':t['trial_wall_seconds']-before['trial_wall_seconds']})
    selected=[]
    for g in a:
        selected.append({'configuration':g,'learning_rate_60':a[g]['learning_rate'],'learning_rate_100':b[g]['learning_rate'],
                         'best_val_60':a[g]['val_accuracy'],'best_epoch_60':a[g]['best_epoch'],
                         'best_val_100':b[g]['val_accuracy'],'best_epoch_100':b[g]['best_epoch'],
                         'gain_percentage_points':100*(b[g]['val_accuracy']-a[g]['val_accuracy']),
                         'best_checkpoint_identical':sha(old/g/'best.pt')==sha(run/g/'best.pt')})
    dest=report/'experiments/baseline_100epochs';dest.mkdir(parents=True,exist_ok=True)
    for name in ['classes.json','completed.json','config.json','environment.json','selected_models.json','trials.json','trials.csv','weights.json','loader_upgrade.json','resume_format.json']:
        if (run/name).exists():shutil.copy2(run/name,dest/name)
    provenance=jread(run/'extension_provenance.json');provenance['source_experiment']='baseline_60epochs'
    provenance['publication_note']='Validation-only exploratory extension; source experiment test metrics remain in baseline_60epochs.'
    jwrite(dest/'extension_provenance.json',provenance)
    for g in a:
        gd=dest/g;gd.mkdir(exist_ok=True)
        for pat in ['*_history.csv','selection.json','*_loader_upgrade.json']:
            for p in (run/g).glob(pat):shutil.copy2(p,gd/p.name)
    plots=dest/'plots';plots.mkdir(exist_ok=True)
    for pat in ['*.png','*.pdf','curve_summary.json']:
        for p in (run/'plots').glob(pat):shutil.copy2(p,plots/p.name)
    csvwrite(report/'tables/60_vs_100_selected.csv',selected)
    csvwrite(report/'tables/60_vs_100_all_trials.csv',detailed)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,g in zip(axes.flat,a):
        rr=rows(run/g/(b[g]['trial_id']+'_history.csv'))
        ax.axvspan(60.5,100,color='#e5f2ed',alpha=.8,label='Additional epochs')
        ax.plot([int(r['epoch']) for r in rr],[100*float(r['val_accuracy']) for r in rr],color='#2679ac',lw=1.4,label='Validation accuracy')
        ax.axvline(60,color='#666666',ls='--',lw=1)
        ax.scatter([a[g]['best_epoch']],[100*a[g]['val_accuracy']],s=110,facecolors='none',edgecolors='#de8b26',linewidths=2,label='Best within 60',zorder=4)
        ax.scatter([b[g]['best_epoch']],[100*b[g]['val_accuracy']],s=130,marker='*',color='#cf4b41',label='Best within 100',zorder=5)
        ax.set(title=f"{g} | lr={b[g]['learning_rate']:g}",xlabel='Epoch',ylabel='Validation accuracy (%)',xlim=(1,100))
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('60 to 100 epochs: validation-selected learning rates',fontsize=16)
    for ext in ['png','pdf']:fig.savefig(report/'figures'/f'extension_60_vs_100.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    total=sum(r['added_recorded_trial_wall_seconds'] for r in detailed)
    summary={'source_test_previously_evaluated':True,'extension_test_evaluated':False,
             'original_histories_preserved':True,'completed_100_trials':sum(t['status']=='completed' for t in trials),
             'skipped_diverged_trials':sum(t['status']=='diverged' for t in trials),
             'additional_training_seconds':sum(t['added_training_seconds'] for t in detailed),
             'additional_validation_seconds':sum(t['added_validation_seconds'] for t in detailed),
             'additional_recorded_trial_wall_seconds':total,'selected_comparison':selected}
    jwrite(report/'tables/60_vs_100_summary.json',summary)
    lines=['# 60 → 100 轮：验证集补充对照','','本表仅比较验证结果；100轮未做测试。原60轮测试结果仍见 final_comparison.md。','',
           '| 配置 | 60轮内最佳验证 | 100轮内最佳验证 | 增益（百分点） | 原/新最佳轮次 |','|---|---:|---:|---:|---:|']
    for r in selected:lines.append(f"| {r['configuration']} | {r['best_val_60']:.2%} | {r['best_val_100']:.2%} | {r['gain_percentage_points']:+.2f} | {r['best_epoch_60']} / {r['best_epoch_100']} |")
    lines += ['',f'新增记录的trial wall time约 {total/3600:.2f} 小时（{total/60:.1f} 分钟）；不含人工暂停、各候选开始前加载等开销。',
              '四组所选学习率仍均为0.001。ResNet34 full最佳权重文件与60轮实验相同（SHA-256核验）；不能据此声称重新测试过100轮模型。',
              '小幅增加的验证最佳值不代表统计显著或测试集改善；从更多轮次取最大值存在选择效应。']
    (report/'tables/60_vs_100_comparison.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    for p in dest.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.csv','.md']:p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    assert all(sha(p)==h for p,h in preserved.items()),'Existing 60-epoch report evidence changed'
    jwrite(report/'file_manifest.json', [{'path':p.relative_to(report).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(report.rglob('*')) if p.is_file() and p.name!='file_manifest.json'])
    print(json.dumps({'added_wall_hours':total/3600,'selected':selected,'original_60_files_unchanged':len(preserved)},indent=2))

if __name__=='__main__':main()
