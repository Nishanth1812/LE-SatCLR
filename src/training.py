import json
import logging
import os
import time
import uuid
import traceback
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from .data import catalog, splits, loader, seed_everything
from .models import SimCLR, Classifier, nt_xent, parameter_counts


def evaluate(model, batches, device, max_batches=0):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for step, (x,y) in enumerate(batches):
            if max_batches and step >= max_batches: break
            logits = model(x.to(device, non_blocking=True))
            if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite validation logits')
            actual.extend(y.tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
    p,r,f,_ = precision_recall_fscore_support(actual,predicted,labels=list(range(10)),average='macro',zero_division=0)
    return dict(accuracy=accuracy_score(actual,predicted),precision=p,recall=r,macro_f1=f), confusion_matrix(actual,predicted,labels=list(range(10)))


def save_checkpoint(path, model, optimizer, scheduler, config, epoch, score, scaler):
    state = dict(model=model.state_dict(),encoder=model.encoder.state_dict(),
                 optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                 scaler=scaler.state_dict(),config=asdict(config),epoch=epoch,score=score)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def train(config, data_root, output_root, encoder_checkpoint=None, tracking_uri=None, commit=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    budget = 'unlabeled' if config.stage == 'simclr' else f'{config.label_percent}pct'
    name = f'{config.stage}_{config.policy}_{budget}_seed42'
    identifier = uuid.uuid4().hex
    root = Path(output_root)
    result_dir = root/'results'/name/identifier
    model_dir = root/'models'/name/identifier
    result_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    log = logging.getLogger(identifier)
    log.setLevel(logging.INFO)
    log.propagate = False
    handlers = [logging.StreamHandler(), logging.FileHandler(result_dir/'training.log',encoding='utf-8')]
    for handler in handlers: log.addHandler(handler)
    start = time.monotonic()

    def event(kind, **fields):
        log.info(json.dumps(dict(event=kind,run_id=identifier,entry_point='train',experiment=name,
                                 elapsed_seconds=round(time.monotonic()-start,2),**fields)))

    try:
        os.environ.setdefault('MLFLOW_HTTP_REQUEST_TIMEOUT','15')
        os.environ.setdefault('MLFLOW_HTTP_REQUEST_MAX_RETRIES','1')
        event('tracking_connecting')
        mlflow.set_tracking_uri(tracking_uri or os.environ['MLFLOW_TRACKING_URI'])
        mlflow.set_experiment('le-satclr')
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(asdict(config))
            mlflow.set_tags({'local_run_id':identifier,
                             'protocol':'transductive' if config.ssl_scope=='all' else 'inductive'})
            event('run_started',mlflow_run_id=run.info.run_id,device=str(device))
            dataset = catalog(data_root)
            split = splits(dataset, root/'results'/'splits_seed42.json')
            mlflow.log_artifact(str(root/'results'/'splits_seed42.json'))
            paired = config.stage == 'simclr'
            indices = (list(range(len(dataset))) if config.ssl_scope=='all' else split['train']) if paired else split[f'labels{config.label_percent}']
            batches = loader(dataset,indices,config.batch_size,True,paired,config.policy,config.workers)
            if len(batches) == 0: raise ValueError('Batch size exceeds available pretraining samples')
            validation = loader(dataset,split['val'],config.batch_size)
            model = SimCLR(config.hidden_dim,config.projection_dim) if paired else Classifier(config.stage=='probe')
            if config.stage in ('probe','finetune'):
                if not encoder_checkpoint: raise ValueError('An SSL checkpoint is required')
                checkpoint = torch.load(encoder_checkpoint,map_location='cpu',weights_only=True)
                if checkpoint['config']['stage'] != 'simclr':
                    raise ValueError('Downstream initialization requires a SimCLR checkpoint')
                if checkpoint['config']['ssl_scope'] != config.ssl_scope:
                    raise ValueError('Checkpoint SSL protocol differs from requested protocol')
                model.encoder.load_state_dict(checkpoint['encoder'],strict=True)
                mlflow.log_param('encoder_checkpoint',str(encoder_checkpoint))
            model.to(device)
            groups = ([{'params':model.encoder.parameters(),'lr':config.encoder_lr},
                       {'params':model.head.parameters(),'lr':config.lr}] if config.stage=='finetune'
                      else [p for p in model.parameters() if p.requires_grad])
            optimizer = torch.optim.AdamW(groups,lr=config.lr,weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epochs)
            mixed = config.amp and device.type=='cuda'
            scaler = torch.amp.GradScaler('cuda',enabled=mixed)
            counts = parameter_counts(model)
            mlflow.log_params({f'parameters_{k}':v for k,v in counts.items()})
            event('dataset_ready',samples=len(indices),batches=len(batches),parameters=counts)
            best = float('-inf')
            history = []
            best_path = model_dir/f'{name}_best.pt'
            for epoch in range(config.epochs):
                epoch_start = time.monotonic()
                model.train()
                total, seen = 0., 0
                for step,(x,y) in enumerate(batches):
                    if config.max_batches and step >= config.max_batches: break
                    x,y = x.to(device, non_blocking=True),y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type,enabled=mixed):
                        if paired:
                            _, z = model(torch.cat((x,y)))
                            loss = nt_xent(z[:len(x)],z[len(x):],config.temperature)
                        else: loss = nn.functional.cross_entropy(model(x),y)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                        event('grad_nonfinite_skipped',epoch=epoch+1,step=step,scale=scaler.get_scale())
                        scaler.update()
                        continue
                    torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=10.,error_if_nonfinite=False)
                    scaler.step(optimizer)
                    scaler.update()
                    total += loss.item()*len(x)
                    seen += len(x)
                    if step % 20 == 0: event('batch_finished',epoch=epoch+1,step=step,loss=loss.item())
                metrics = {'train_loss':total/seen,'learning_rate':optimizer.param_groups[0]['lr']}
                if paired:
                    # Label-free retrieval accuracy: match the two views of each validation image.
                    model.eval()
                    correct, count = 0,0
                    with torch.no_grad():
                        for step,(a,b) in enumerate(loader(dataset,split['val'],config.batch_size,True,True,config.policy)):
                            if config.max_batches and step >= config.max_batches: break
                            _,z = model(torch.cat((a,b)).to(device, non_blocking=True))
                            n = len(a)
                            similarity = z[:n] @ z[n:].T
                            target = torch.arange(n,device=device)
                            correct += ((similarity.argmax(1)==target).sum()+(similarity.argmax(0)==target).sum()).item()
                            count += 2*n
                    metrics['val_pair_accuracy'] = correct/count
                    score = -metrics['train_loss']
                else:
                    val,_ = evaluate(model,validation,device,config.max_batches)
                    metrics.update({f'val_{k}':v for k,v in val.items()})
                    score = val['accuracy']
                metrics['epoch_seconds'] = time.monotonic() - epoch_start
                metrics['samples_per_second'] = seen / metrics['epoch_seconds']
                if device.type == 'cuda':
                    metrics['gpu_peak_memory_mb'] = torch.cuda.max_memory_allocated()/1024**2
                scheduler.step()
                if score > best:
                    best = score
                    save_checkpoint(best_path,model,optimizer,scheduler,config,epoch,best,scaler)
                    event('best_model_saved',epoch=epoch+1,score=best,path=str(best_path))
                history.append(dict(epoch=epoch+1,**metrics))
                (result_dir/'history.json').write_text(json.dumps(history),encoding='utf-8')
                mlflow.log_metrics(metrics,step=epoch+1)
                event('epoch_finished',epoch=epoch+1,**metrics)
                if commit: commit()
            restored = torch.load(best_path,map_location=device,weights_only=True)
            model.load_state_dict(restored['model'])
            summary = dict(config=asdict(config),counts=counts,checkpoint=str(best_path),mlflow_run_id=run.info.run_id)
            if not paired:
                test, matrix = evaluate(model,loader(dataset,split['test'],config.batch_size),device,config.max_batches)
                summary['test'] = test
                mlflow.log_metrics({f'test_{k}':v for k,v in test.items()})
                np.savetxt(result_dir/'confusion_matrix.csv',matrix,delimiter=',',fmt='%d')
            (result_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            event('run_finished',checkpoint=str(best_path))
            mlflow.log_artifacts(str(result_dir),artifact_path='results')
            mlflow.log_artifact(str(best_path),artifact_path='models')
            return summary
    except Exception:
        event('run_failed',traceback=traceback.format_exc())
        raise
    finally:
        for handler in handlers:
            handler.flush()
            handler.close()
            log.removeHandler(handler)
        if commit: commit()
