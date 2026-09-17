"""Generate comparison figures from completed runs: python -m src.report --help."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def results_figure(results_markdown, figure):
    text = Path(results_markdown).read_text(encoding='utf-8')
    if 'Transductive SimCLR pretraining (`ssl_scope=all`)' not in text or 'seed 42' not in text:
        raise ValueError('Expected transductive ssl_scope=all results with seed 42')
    section = text.split('## Primary result — test accuracy at fixed label budgets\n',1)[1].split('\n## ',1)[0]
    methods = {
        'Supervised ResNet-18 from scratch': 'Supervised from scratch',
        'SimCLR + Linear probe (frozen)': 'SimCLR + frozen probe',
        'SimCLR + Fine-tuning': 'SimCLR + fine-tuning',
    }
    values = {}
    for line in section.splitlines():
        cells = [cell.strip().replace('**','') for cell in line.strip().strip('|').split('|')]
        if cells[0] in methods:
            if len(cells) != 3 or cells[0] in values:
                raise ValueError('Expected one result per method at two label budgets')
            scores = [float(value) for value in cells[1:]]
            if not all(np.isfinite(score) and 0 <= score <= 1 for score in scores):
                raise ValueError('Test accuracy must be finite and between zero and one')
            values[cells[0]] = np.asarray(scores)*100
    if values.keys() != methods.keys():
        raise ValueError('Missing measured method in primary results table')
    dest = Path(figure)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':11}):
        fig,ax = plt.subplots(figsize=(10,6))
        x = np.arange(2)
        for i,((method,label),color) in enumerate(zip(methods.items(),('#65758b','#3478b8','#18866b'))):
            bars = ax.bar(x+(i-1)*0.24,values[method],width=0.24,label=label,color=color)
            ax.bar_label(bars,labels=[f'{score:.2f}%' for score in values[method]],padding=4,fontsize=10)
        ax.set(xticks=x,xticklabels=['1% (270 labels)','10% (2,700 labels)'],
               xlabel='Label budget (% of the full EuroSAT dataset)',ylabel='Test accuracy (%)',ylim=(0,100))
        ax.set_axisbelow(True)
        ax.grid(axis='y',alpha=0.2)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='lower right',framealpha=1)
        fig.suptitle('EuroSAT: measured label efficiency',fontsize=17,fontweight='bold',y=0.97)
        ax.set_title('Single seed (42) · Transductive SimCLR · ssl_scope=all',fontsize=11,pad=15)
        fig.text(0.5,0.065,'Source: RESULTS.md · Standard augmentation · No multi-seed uncertainty estimates',ha='center',fontsize=10)
        fig.text(0.5,0.03,'SSL includes unlabeled validation/test images; not an inductive generalization result.',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,0.1,1,0.93))
        fig.savefig(dest,dpi=160)
        plt.close(fig)
    return dest


def report(output_root, data_root, ssl_checkpoint, finetuned_checkpoint, max_samples=0):
    import torch
    from PIL import Image
    from sklearn.metrics.pairwise import cosine_similarity
    from umap import UMAP

    from .data import catalog, loader, seed_everything
    from .models import encoder

    seed_everything()
    root = Path(output_root)
    dest = root/'results'/'report'
    dest.mkdir(parents=True,exist_ok=True)
    rows = []
    for path in sorted((root/'results').glob('*/*/summary.json'),key=lambda p:p.stat().st_mtime):
        summary = json.loads(path.read_text())
        if summary['config']['max_batches']: continue
        if 'test' in summary:
            rows.append(dict(method=summary['config']['stage'],policy=summary['config']['policy'],protocol=summary['config']['ssl_scope'],
                             label_percent=summary['config']['label_percent'],**summary['test'],
                             run_id=summary['mlflow_run_id']))
        history = json.loads((path.parent/'history.json').read_text())
        fig,axes = plt.subplots(1,2,figsize=(10,4))
        axes[0].plot([h['epoch'] for h in history],[h['train_loss'] for h in history])
        axes[0].set(title='Training loss',xlabel='Epoch')
        metric = 'val_accuracy' if 'val_accuracy' in history[0] else 'val_pair_accuracy'
        axes[1].plot([h['epoch'] for h in history],[h[metric] for h in history])
        axes[1].set(title=metric,xlabel='Epoch')
        fig.tight_layout(); fig.savefig(dest/f'{path.parent.name}_curves.png'); plt.close(fig)
        matrix_path = path.parent/'confusion_matrix.csv'
        if matrix_path.exists():
            fig,ax = plt.subplots(figsize=(9,8))
            ax.imshow(np.loadtxt(matrix_path,delimiter=','))
            ax.set(xlabel='Predicted class index',ylabel='True class index')
            fig.savefig(dest/f'{path.parent.name}_confusion.png'); plt.close(fig)
    if rows:
        with (dest/'comparison.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        latest = {(r['method'],r['policy'],r['protocol'],r['label_percent']):r for r in rows}
        table = ['Latest completed run per method, policy, protocol and budget. All runs retained in comparison.csv.', '',
                 '| Method / policy / protocol | Accuracy @ 1% | Accuracy @ 10% |',
                 '|---|---:|---:|']
        for method,policy,protocol in sorted({key[:3] for key in latest}):
            values = [str(latest[(method,policy,protocol,budget)]['accuracy']) if (method,policy,protocol,budget) in latest else '—' for budget in (1,10)]
            table.append(f"| {method} / {policy} / {protocol} | {' | '.join(values)} |")
        (dest/'comparison.md').write_text('\n'.join(table),encoding='utf-8')
        fig,ax = plt.subplots()
        for method in ('baseline','probe','finetune'):
            for policy in ('standard','satellite'):
                for protocol in ('all','train'):
                    selected = sorted([r for r in latest.values() if r['method']==method and r['policy']==policy and r['protocol']==protocol],key=lambda r:r['label_percent'])
                    if selected: ax.plot([r['label_percent'] for r in selected],[r['accuracy'] for r in selected],marker='o',label=f'{method}/{policy}/{protocol}')
        ax.set(xlabel='Labels (% of entire dataset)',ylabel='Test accuracy'); ax.legend()
        fig.savefig(dest/'label_efficiency.png'); plt.close(fig)
    dataset = catalog(data_root)
    split = json.loads((root/'results'/'splits_seed42.json').read_text())
    indices = split['val'] + split['test']
    if max_samples: indices = indices[:max_samples]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for name,checkpoint in [('random',None),('ssl',ssl_checkpoint),('finetuned',finetuned_checkpoint)]:
        seed_everything()
        model = encoder()
        if checkpoint: model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['encoder'])
        model.to(device).eval()
        features,labels = [],[]
        with torch.no_grad():
            for x,y in loader(dataset,indices,128):
                features.append(model(x.to(device)).cpu().numpy()); labels.extend(y.tolist())
        features = np.concatenate(features)
        if not np.isfinite(features).all(): raise FloatingPointError('Invalid embeddings')
        np.savez_compressed(dest/f'{name}_embeddings.npz',features=features,labels=labels,indices=indices)
        points = UMAP(n_components=2,random_state=42,n_jobs=1).fit_transform(features)
        fig,ax = plt.subplots(figsize=(8,6))
        for label,classname in enumerate(dataset.classes):
            mask = np.asarray(labels)==label
            ax.scatter(points[mask,0],points[mask,1],s=5,label=classname)
        ax.set_title(f'{name}: encoder features (UMAP)'); ax.legend(fontsize=6)
        fig.savefig(dest/f'{name}_umap.png'); plt.close(fig)
        scores = cosine_similarity(features[:1],features)[0]
        scores[0] = -np.inf  # Exclude the query itself.
        nearest = [0] + scores.argsort()[-5:][::-1].tolist()
        fig,axes = plt.subplots(1,len(nearest),figsize=(12,3))
        for ax,index in zip(axes,nearest):
            with Image.open(dataset.samples[indices[index]][0]) as image: ax.imshow(image.convert('RGB'))
            ax.set_title('Query' if index==0 else dataset.classes[labels[index]],fontsize=7); ax.axis('off')
        fig.savefig(dest/f'{name}_retrieval.png'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root',default='outputs')
    parser.add_argument('--data-root',default='Dataset/EuroSAT_RGB')
    parser.add_argument('--ssl-checkpoint')
    parser.add_argument('--finetuned-checkpoint')
    parser.add_argument('--max-samples',type=int,default=0)
    parser.add_argument('--results-markdown',help='Plot the measured primary table without datasets or checkpoints')
    parser.add_argument('--figure',default='docs/results.png')
    args = vars(parser.parse_args())
    markdown = args.pop('results_markdown')
    figure = args.pop('figure')
    if markdown:
        results_figure(markdown,figure)
    elif not args['ssl_checkpoint'] or not args['finetuned_checkpoint']:
        parser.error('--ssl-checkpoint and --finetuned-checkpoint are required unless --results-markdown is used')
    else:
        report(**args)
