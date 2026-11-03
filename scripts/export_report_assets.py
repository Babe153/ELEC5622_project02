"""Export report evidence without copying datasets or model checkpoints; no training/test inference."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report'
GROUPS=['alexnet_partial','alexnet_full','resnet34_partial','resnet34_full']
NAMES=['AlexNet partial','AlexNet full','ResNet34 partial','ResNet34 full']

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)
    for experiment in ['baseline_30epochs','baseline_60epochs']:
        run=ROOT/'results'/experiment
        dest=OUT/'experiments'/experiment;dest.mkdir(parents=True,exist_ok=True)
        # Explicit allowlist: never include .pt/.pth, absolute-path logs, or original data.
        for name in ['classes.json','completed.json','config.json','environment.json','selected_models.json','trials.json','trials.csv','weights.json','test_results.json','test_results.csv','loader_upgrade.json','resume_format.json']:
            if (run/name).exists(): shutil.copy2(run/name,dest/name)
        for g in GROUPS:
            gd=dest/g;gd.mkdir(exist_ok=True)
            for pat in ['*_history.csv','selection.json','confusion_matrix.csv','test_predictions.csv','*_loader_upgrade.json']:
                for p in (run/g).glob(pat): shutil.copy2(p,gd/p.name)
        pd=dest/'plots';pd.mkdir(exist_ok=True)
        for pat in ['*.png','*.pdf','curve_summary.json']:
            for p in (run/'plots').glob(pat): shutil.copy2(p,pd/p.name)
    run=ROOT/'results/baseline_60epochs'
    selected=json.loads((run/'selected_models.json').read_text())
    result=json.loads((run/'test_results.json').read_text())
    classes=json.loads((run/'classes.json').read_text())
    tables=OUT/'tables';tables.mkdir(exist_ok=True)
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    final=[];classrows=[];matrices=[];growth=[]
    for r in result:
        g=r['configuration'];preds=read_csv(run/g/'test_predictions.csv')
        assert len(preds)==2720 and len({p['image'] for p in preds})==2720
        matrix=np.array([[sum(p['true_class']==a and p['predicted_class']==b for p in preds) for b in classes] for a in classes])
        assert matrix.tolist()==[[int(row[c]) for c in classes] for row in read_csv(run/g/'confusion_matrix.csv')]
        correct=int(np.trace(matrix));assert abs(correct/2720-r['test_accuracy'])<1e-12
        matrices.append(matrix)
        f1=[]
        for i,c in enumerate(classes):
            tp=int(matrix[i,i]);support=int(matrix[i].sum());predicted=int(matrix[:,i].sum())
            precision=tp/predicted if predicted else 0;recall=tp/support
            score=2*precision*recall/(precision+recall) if precision+recall else 0
            f1.append(score)
            classrows.append(dict(configuration=g,cell_class=c,support=support,precision=precision,recall=recall,f1=score))
        final.append(dict(configuration=g,learning_rate=r['learning_rate'],best_epoch=r['best_epoch'],validation_accuracy=selected[g]['val_accuracy'],test_accuracy=r['test_accuracy'],correct=correct,errors=2720-correct,trainable_parameters=r['trainable_parameters'],training_minutes_to_best=r['training_seconds_to_best']/60,selected_trial_60_epoch_training_minutes=r['selected_trial_training_seconds']/60,group_tuning_wall_minutes=r['group_tuning_wall_seconds']/60,macro_f1=sum(f1)/6))
        # Controlled descriptive comparison within the SAME 60-epoch experiment.
        for t in json.loads((run/'trials.json').read_text()):
            if t['group']!=g or t['status']!='completed':continue
            rows=read_csv(run/g/(t['trial_id']+'_history.csv'))
            best=lambda rr:max(rr,key=lambda x:(float(x['val_accuracy']),-float(x['val_loss'])))
            b30=best([x for x in rows if int(x['epoch'])<=30]);b60=best(rows)
            growth.append(dict(configuration=g,learning_rate=t['learning_rate'],best_val_first30=float(b30['val_accuracy']),best_epoch_first30=int(b30['epoch']),best_val_first60=float(b60['val_accuracy']),best_epoch_first60=int(b60['epoch']),gain_percentage_points=100*(float(b60['val_accuracy'])-float(b30['val_accuracy']))))
    write_csv(tables/'final_comparison.csv',final);write_csv(tables/'per_class_test_metrics.csv',classrows);write_csv(tables/'within_run_30_vs_60.csv',growth)
    labels={int(r['Image ID']):r['Image class'].strip() for r in read_csv(ROOT/'datasets/gt_training.csv')}
    counts=[]
    for c in classes:
        counts.append({'class':c,**{split:sum(labels[int(p.stem)]==c for p in (ROOT/'datasets'/split).glob('*.png')) for split in ['training','validation','test']}})
    write_csv(tables/'class_distribution.csv',counts)
    fig,axes=plt.subplots(2,2,figsize=(13,11),layout='constrained')
    for ax,m,title in zip(axes.flat,matrices,NAMES):
        norm=m/m.sum(axis=1,keepdims=True)
        im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        ax.set_title(title);ax.set_xticks(range(6),classes,rotation=40,ha='right');ax.set_yticks(range(6),classes)
        ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
        for i in range(6):
            for j in range(6):ax.text(j,i,str(m[i,j]),ha='center',va='center',color='white' if norm[i,j]>.55 else 'black',fontsize=9)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label='Fraction within true class')
    fig.suptitle('Final test confusion matrices: labels are counts; color is row-normalized')
    for ext in ['png','pdf']:fig.savefig(figs/f'test_confusion_matrices.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,key,scale,title in zip(axes,['test_accuracy','trainable_parameters','selected_trial_60_epoch_training_minutes'],[100,1e-6,1],['Test accuracy (%)','Trainable parameters (million)','Selected trial training time (min, 60 epochs)']):
        vals=[r[key]*scale for r in final];bars=ax.bar(range(4),vals,color=['#6199c6','#246496','#71b79d','#21755a'])
        ax.set_xticks(range(4),['AlexNet\npartial','AlexNet\nfull','ResNet34\npartial','ResNet34\nfull']);ax.set_title(title,fontsize=10)
        ax.set_ylim(0,max(vals)*1.17);ax.bar_label(bars,fmt='%.2f',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(figs/f'accuracy_parameters_time.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    lines=['# Verified final comparison','','Accuracy is a fraction in CSV and a percentage below.','', '| Configuration | Best epoch | Validation | Test | Errors | Parameters | Training to best (min) | Training 60 epochs (min) | All-LR tuning wall time (min) |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in final:
        lines.append(f"| {r['configuration']} | {r['best_epoch']} | {r['validation_accuracy']:.2%} | {r['test_accuracy']:.2%} | {r['errors']} | {r['trainable_parameters']:,} | {r['training_minutes_to_best']:.2f} | {r['selected_trial_60_epoch_training_minutes']:.2f} | {r['group_tuning_wall_minutes']:.2f} |")
    (tables/'final_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Preserve portable byte hashes and avoid platform-dependent line endings.
    for p in OUT.rglob('*'):
        if p.is_file() and (p.suffix in ['.md','.json','.csv'] or p.name=='.gitattributes'):
            p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    (OUT/'file_manifest.json').write_text(json.dumps([{'path':str(p.relative_to(OUT)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='file_manifest.json'],indent=2),encoding='utf-8')
    manifest_path=OUT/'file_manifest.json'
    manifest_path.write_bytes(manifest_path.read_bytes().replace(b'\r\n',b'\n'))
    print('\n'.join(lines))
    print('Verified test predictions, confusion matrices and accuracy for all four models.')

if __name__=='__main__':main()
