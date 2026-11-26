"""PeakCMA training and seed-matched baseline pretraining."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_provider.peak_data import PeakDataset, SETTINGS
from models.backbone import CrossModalBackbone
from models.PeakCMA import PeakCMA
from utils.peak import soft_labels, peak_loss, diagnostics, select_threshold


def parse_args():
    p = argparse.ArgumentParser(description='PeakCMA reproduction')
    p.add_argument('--data_path', choices=list(SETTINGS), default='Weather')
    p.add_argument('--root_path', default='./dataset')
    p.add_argument('--embedding_root', default='./Embeddings/PeakCMA')
    p.add_argument('--columns', default='')
    p.add_argument('--peak_columns', default='')
    p.add_argument('--mode', choices=['baseline','peak'], default='peak')
    p.add_argument('--baseline_checkpoint')
    p.add_argument('--llm_model_path', default='gpt2')
    p.add_argument('--prompt_batch_size', type=int, default=4)
    p.add_argument('--embedding_batch_size',type=int,default=1)
    p.add_argument('--embedding_precision',choices=['float32','float16','bfloat16'],default='float32')
    p.add_argument('--embedding_progress',default=None)
    p.add_argument('--resume',default=None,help='Epoch checkpoint path, or auto to use last_training.pth')
    p.add_argument('--preload_embeddings',action='store_true',help='Cache all selected embeddings in host RAM')
    p.add_argument('--window_manifest', default=None, help='Retain valid preselected windows before uniformly adding more')
    p.add_argument('--min_epochs',type=int,default=0)
    p.add_argument('--selection_metric',choices=['mse','joint','event_f1'],default='mse')
    p.add_argument('--max_validation_mse',type=float,default=None)
    for split in ('train','val','test'):
        p.add_argument('--max_'+split+'_samples', type=int, default=0,
                       help='0 uses every window; positive limits evenly spaced windows after splitting')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    for name,default in [('seq_len',None),('pred_len',None),('channel',32),('head',8),('e_layer',1),('d_layer',1),('d_llm',768),('batch_size',32),('epochs',100),('num_workers',10),('seed',2024),('freeze_epochs',4),('es_patience',50)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('dropout_n',.1),('learning_rate',1e-4),('backbone_lr',1e-5),('weight_decay',.001),('omega',5.),('lambda_loc',.1),('lambda_peak',.5)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--save',default='./Results/PeakCMA')
    args=p.parse_args()
    args.seq_len=args.seq_len or SETTINGS[args.data_path][0]
    args.pred_len=args.pred_len or SETTINGS[args.data_path][1]
    return args


@torch.no_grad()
def evaluate(model,loader,device,collect=False):
    model.eval(); total=absolute=count=0; arrays=[[],[],[],[]]
    for x,y,mark,u,emb in loader:
        x,y,mark,emb=[v.to(device).float() for v in (x,y,mark,emb)]
        if isinstance(model,PeakCMA):
            pred,aux=model(x,mark,emb,return_aux=True); score=aux['score']
        else: pred=model(x,mark,emb); score=pred
        total+=(pred-y).square().sum().item(); absolute+=(pred-y).abs().sum().item(); count+=y.numel()
        if collect:
            for dest,value in zip(arrays,(y,pred,score,u)): dest.append(value.cpu().numpy())
    return total/count,absolute/count,[np.concatenate(a) for a in arrays] if collect else None


def event_report(arrays,channels,theta,w,separation):
    y,pred,score,u=arrays
    return diagnostics(y,pred,score,u,channels,theta,w,separation)


def main():
    args=parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    datasets=[PeakDataset(args,s) for s in ('train','val','test')]
    loaders=[DataLoader(d,batch_size=args.batch_size,shuffle=False,drop_last=False,num_workers=args.num_workers) for d in datasets]
    kwargs=dict(device=args.device,channel=args.channel,num_nodes=len(datasets[0].columns),seq_len=args.seq_len,pred_len=args.pred_len,dropout_n=args.dropout_n,d_llm=args.d_llm,e_layer=args.e_layer,d_layer=args.d_layer,head=args.head)
    if args.mode=='peak':
        if not args.baseline_checkpoint: raise ValueError('Supply --baseline_checkpoint from the matched baseline run')
        model=PeakCMA(datasets[0].channels,**kwargs).to(args.device)
        checkpoint=torch.load(args.baseline_checkpoint,map_location=args.device,weights_only=False)
        if 'model' in checkpoint:
            if checkpoint.get('preprocessing_version') != datasets[0].preprocessing_version:
                raise ValueError('Baseline preprocessing mismatch: retrain the baseline after data cleaning')
            if checkpoint['normalisation'] != datasets[0].normalisation:
                raise ValueError('Baseline normalisation mismatch')
            for key in ('seed','data_path','seq_len','pred_len'):
                if checkpoint['args'][key]!=getattr(args,key): raise ValueError('Baseline mismatch: '+key)
            if checkpoint['columns']!=datasets[0].columns: raise ValueError('Baseline column order mismatch')
            model.load_baseline(checkpoint['model'])
        else: model.load_baseline(checkpoint)
        new=[v for k,v in model.named_parameters() if not k.startswith('backbone.')]
        groups=[{'params':model.backbone.parameters(),'lr':args.backbone_lr},{'params':new,'lr':args.learning_rate}]
    else:
        model=CrossModalBackbone(**kwargs).to(args.device); groups=[{'params':model.parameters(),'lr':args.learning_rate}]
    optimizer=torch.optim.AdamW(groups,weight_decay=args.weight_decay)
    def schedule(group):
        start=args.freeze_epochs if args.mode=='peak' and group==0 else 0
        rate=groups[group]['lr']
        def factor(epoch):
            progress=max(0,epoch-start)/max(1,args.epochs-start)
            return 1e-6/rate+(1-1e-6/rate)*(1+np.cos(np.pi*min(1.,progress)))/2
        return factor
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,[schedule(i) for i in range(len(groups))])
    out=Path(args.save)/args.data_path/f'{args.seq_len}_{args.pred_len}'/str(args.seed)/args.mode
    out.mkdir(parents=True,exist_ok=True); best=float('inf'); stale=0
    manifest = {s: {'windows':len(d), 'selected_windows':d.samples,
                    'cache':str(d.cache), 'columns':d.columns,'preprocessing_version':d.preprocessing_version,
                    'sentinel_counts':{k:int(v) for k,v in d.sentinel_counts.items()}}
                for s,d in zip(('train','val','test'),datasets)}
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    _,_,_,w,separation,_=SETTINGS[args.data_path]
    history=[]; start_epoch=0
    if args.resume:
        resume_path=out/'last_training.pth' if args.resume=='auto' else Path(args.resume)
        if resume_path.exists():
            resumed=torch.load(resume_path,map_location=args.device,weights_only=False)
            if resumed.get('dataset_caches') != [str(d.cache) for d in datasets]:
                raise ValueError('Resume data fingerprint mismatch')
            if not (out/'best_model.pth').exists():
                raise FileNotFoundError('Keep best_model.pth alongside last_training.pth when resuming')
            for key,value in vars(args).items():
                if key not in ('resume','device','num_workers','preload_embeddings') and resumed['args'].get(key)!=value:
                    raise ValueError('Resume configuration mismatch: '+key)
            model.load_state_dict(resumed['model']); optimizer.load_state_dict(resumed['optimizer'])
            scheduler.load_state_dict(resumed['scheduler'])
            start_epoch=resumed['epoch']; best=resumed['best']; stale=resumed['stale']; history=resumed['history']
            random.setstate(resumed['python_rng']); np.random.set_state(resumed['numpy_rng'])
            torch.set_rng_state(resumed['torch_rng'].cpu())
            if torch.cuda.is_available() and resumed.get('cuda_rng') is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in resumed['cuda_rng']])
            print(f'Resumed from epoch {start_epoch}',flush=True)
        elif args.resume!='auto': raise FileNotFoundError(resume_path)
    end_epoch=start_epoch if stale>=args.es_patience and start_epoch>=args.min_epochs else args.epochs
    for epoch in range(start_epoch,end_epoch):
        model.train()
        if args.mode=='peak': model.freeze_backbone(epoch<args.freeze_epochs)
        train_total = 0.
        totals=dict(mse=0.,loc=0.,peak=0.)
        for x,y,mark,u,emb in loaders[0]:
            x,y,mark,emb=[v.to(args.device).float() for v in (x,y,mark,emb)]
            optimizer.zero_grad()
            if args.mode=='peak':
                pred,aux=model(x,mark,emb,return_aux=True)
                labels=torch.cat([soft_labels(v[None],threshold.numpy(),datasets[0].channels,w,separation) for v,threshold in zip(y,u)])
                loss,parts=peak_loss(pred,y,aux['logits'],labels,model.peak_mask,args.omega,args.lambda_loc,args.lambda_peak,True)
            else:
                loss=(model(x,mark,emb)-y).square().mean(); parts={'mse':loss}
            for key,value in parts.items(): totals[key]+=value.item()*len(y)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
            train_total += loss.item()*len(y)
        val,mae,_=evaluate(model,loaders[1],args.device)
        validation_parts={}
        if args.mode=='peak':
            with torch.no_grad():
                sums=dict(mse=0.,loc=0.,peak=0.)
                positive_score=positive_mass=negative_score=negative_mass=0.
                for x,y,mark,u,emb in loaders[1]:
                    x,y,mark,emb=[v.to(args.device).float() for v in (x,y,mark,emb)]
                    pred,aux=model(x,mark,emb,True)
                    labels=torch.cat([soft_labels(v[None],threshold.numpy(),datasets[0].channels,w,separation) for v,threshold in zip(y,u)])
                    active=model.peak_mask[None,None,:]
                    positive_score+=(aux['score']*labels*active).sum().item()
                    positive_mass+=(labels*active).sum().item()
                    negative=(labels==0)*active
                    negative_score+=(aux['score']*negative).sum().item()
                    negative_mass+=negative.sum().item()
                    _,parts=peak_loss(pred,y,aux['logits'],labels,model.peak_mask,args.omega,args.lambda_loc,args.lambda_peak,True)
                    for key,value in parts.items(): sums[key]+=value.item()*len(y)
                validation_parts={k:v/len(datasets[1]) for k,v in sums.items()}
                validation_parts['positive_score']=positive_score/max(1e-8,positive_mass)
                validation_parts['background_score']=negative_score/max(1e-8,negative_mass)
        criterion=val if args.selection_metric=='mse' or args.mode=='baseline' else (validation_parts['mse']+args.lambda_loc*validation_parts['loc']+args.lambda_peak*validation_parts['peak'])
        if args.mode=='peak' and args.selection_metric=='event_f1':
            _,_,va=evaluate(model,loaders[1],args.device,True)
            chosen,event_counts=select_threshold(va[0],va[2],va[3],datasets[0].channels,w,separation)
            validation_parts['event_f1']=event_counts['f1']
            validation_parts['event_threshold']=chosen
            permitted=args.max_validation_mse is None or val<=args.max_validation_mse
            criterion=-event_counts['f1'] if permitted else float('inf')
        record=dict(epoch=epoch+1,train={k:v/len(datasets[0]) for k,v in totals.items()},validation=validation_parts or {'mse':val},selection_value=criterion if np.isfinite(criterion) else None,learning_rates=[g['lr'] for g in optimizer.param_groups])
        history.append(record)
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(f'Epoch {epoch+1}: train loss={train_total/len(datasets[0]):.6f}, validation MSE={val:.6f}, MAE={mae:.6f}, backbone_frozen={args.mode=="peak" and epoch<args.freeze_epochs}',flush=True)
        if validation_parts:
            print('  Validation components: '+json.dumps(validation_parts),flush=True)
        if criterion<best:
            best=criterion; stale=0
            best_temporary=out/'best_model.pth.tmp'
            torch.save(dict(model=model.state_dict(),args=vars(args),columns=datasets[0].columns,
                            normalisation=datasets[0].normalisation,preprocessing_version=datasets[0].preprocessing_version,epoch=epoch+1),best_temporary)
            best_temporary.replace(out/'best_model.pth')
        else: stale+=1
        scheduler.step()
        last=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                  args=vars(args),epoch=epoch+1,best=best,stale=stale,history=history,
                  python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                  dataset_caches=[str(d.cache) for d in datasets])
        temporary=out/'last_training.pth.tmp'
        torch.save(last,temporary); temporary.replace(out/'last_training.pth')
        if stale>=args.es_patience and epoch+1>=args.min_epochs: break
    if not np.isfinite(best):
        raise RuntimeError('No eligible checkpoint satisfied the validation constraint; inspect history.json')
    model.load_state_dict(torch.load(out/'best_model.pth',map_location=args.device,weights_only=False)['model'])
    theta=None
    if args.mode=='peak':
        _,_,arrays=evaluate(model,loaders[1],args.device,True)
        theta,_=select_threshold(arrays[0],arrays[2],arrays[3],datasets[0].channels,w,separation)
    mse,mae,arrays=evaluate(model,loaders[2],args.device,True)
    np.savez_compressed(out/'test_predictions.npz',target=arrays[0],prediction=arrays[1],score=arrays[2],thresholds=arrays[3])
    report=dict(mse=mse,mae=mae,theta=theta)
    report.update(event_report(arrays,datasets[0].channels,theta,w,separation))
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report)


if __name__=='__main__': main()
