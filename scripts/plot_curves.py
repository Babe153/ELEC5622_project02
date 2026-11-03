"""Plot saved epoch metrics only; never starts training or evaluates the test set."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ['alexnet_partial', 'alexnet_full', 'resnet34_partial', 'resnet34_full']
TITLES = ['AlexNet / partial', 'AlexNet / full', 'ResNet34 / partial', 'ResNet34 / full']
COLORS = ['#c76327', '#2476b4', '#239277']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'results/baseline_60epochs')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = (args.output_dir or run / 'plots').resolve()
    out.mkdir(parents=True, exist_ok=True)
    trials = json.loads((run / 'trials.json').read_text(encoding='utf-8'))
    selected = json.loads((run / 'selected_models.json').read_text(encoding='utf-8'))
    histories = {}
    for t in trials:
        path = run / t['group'] / (t['trial_id'] + '_history.csv')
        if path.exists():
            with path.open(newline='', encoding='utf-8-sig') as f:
                histories[(t['group'], t['trial_id'])] = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    max_epoch = max((r['epoch'] for rows in histories.values() for r in rows), default=1)
    min_val = min((100*r['val_accuracy'] for rows in histories.values() for r in rows), default=0)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white'})
    def style(ax, label):
        ax.set_xlabel('Epoch'); ax.set_ylabel(label); ax.set_xlim(1, max(2, max_epoch))
        ax.grid(alpha=.2)
    def save(fig, name):
        fig.savefig(out / (name + '.png'), dpi=170, bbox_inches='tight')
        fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
        plt.close(fig)
    overview, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    overview.suptitle(f'Validation accuracy / {int(max_epoch)} epochs / all learning rates', fontsize=16)
    winners, winaxes = plt.subplots(4, 2, figsize=(12, 14), layout='constrained')
    winners.suptitle(f'Selected learning rates / {int(max_epoch)} epochs / training and validation', fontsize=16)
    summary = []
    for gi, (group, title) in enumerate(zip(GROUPS, TITLES)):
        ax = axes.flat[gi]
        ax.set_title(title); style(ax, 'Validation accuracy (%)'); ax.set_ylim(max(0, min_val - 3), 100)
        group_trials = [t for t in trials if t['group'] == group]
        if not group_trials:
            ax.set_visible(False)
            for a in winaxes[gi]: a.set_visible(False)
            continue
        detail, dax = plt.subplots(len(group_trials), 2, figsize=(12, 10), layout='constrained', squeeze=False)
        detail.suptitle(title + ' / all trials', fontsize=16)
        for ti, t in enumerate(group_trials):
            rows = histories.get((group, t['trial_id']), [])
            lr = t['learning_rate']
            if not rows:
                for a in dax[ti]:
                    a.text(.5, .5, f'lr={lr:g}: {t["status"]}\nNo completed epoch; no curve available', ha='center', va='center', transform=a.transAxes)
                    a.set_axis_off()
                ax.plot([], [], color=COLORS[ti % len(COLORS)], label=f'lr={lr:g}: {t["status"]} (no epoch)')
                continue
            trial_label = f'lr={lr:g}' + (f' (diverged after {len(rows)} epochs)' if t['status'] == 'diverged' else '')
            epochs = [r['epoch'] for r in rows]
            vals = [100*r['val_accuracy'] for r in rows]
            ax.plot(epochs, vals, color=COLORS[ti % len(COLORS)], label=trial_label, linewidth=1.5)
            best = max(rows, key=lambda r: (r['val_accuracy'], -r['val_loss']))
            ax.scatter([best['epoch']], [best['val_accuracy']*100], color=COLORS[ti % len(COLORS)], marker='*', s=95, zorder=3)
            for col, (metric, factor, label) in enumerate([('accuracy', 100, 'Accuracy (%)'), ('loss', 1, 'Cross-entropy loss')]):
                a = dax[ti, col]
                style(a, label); a.set_title(f'{trial_label} / {label}')
                for split, color in [('train', '#2476b4'), ('val', '#c76327')]:
                    a.plot(epochs, [factor*r[split+'_'+metric] for r in rows], label=split, color=color)
                a.axvline(best['epoch'], color='gray', alpha=.5, linestyle=':', label=f'Best val epoch {int(best["epoch"])}')
                a.legend(fontsize=8)
                if t['trial_id'] == selected[group]['trial_id']:
                    wa = winaxes[gi, col]
                    style(wa, label); wa.set_title(f'{title} / lr={lr:g}')
                    for split, color in [('train', '#2476b4'), ('val', '#c76327')]:
                        wa.plot(epochs, [factor*r[split+'_'+metric] for r in rows], label=split, color=color)
                    wa.axvline(best['epoch'], color='gray', alpha=.5, linestyle=':')
                    wa.legend(fontsize=8)
            prev, last = rows[-10:-5], rows[-5:]
            avg = lambda rs, key: sum(r[key] for r in rs)/len(rs) if rs else None
            summary.append({'group':group, 'lr':lr, 'best_epoch':int(best['epoch']), 'best_val_pct':100*best['val_accuracy'], 'previous_window_epochs':[int(r['epoch']) for r in prev], 'last_window_epochs':[int(r['epoch']) for r in last], 'val_acc_mean_previous':avg(prev,'val_accuracy'), 'val_acc_mean_last':avg(last,'val_accuracy'), 'val_loss_mean_previous':avg(prev,'val_loss'), 'val_loss_mean_last':avg(last,'val_loss')})
        ax.legend(fontsize=8, loc='lower right')
        save(detail, group + '_curves')
    save(overview, 'validation_comparison')
    save(winners, 'selected_training_curves')
    (out / 'curve_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Plots saved: {out}')

if __name__ == '__main__':
    main()
