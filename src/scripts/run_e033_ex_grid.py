"""E033: grouped multi-transition training with _ex sources for one PCA/K point."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import MiniBatchKMeans

from src.approaches.population_transformer.nonlinear_pca import ApproximateKernelPCA, GroupTransitionTransformer
from src.m0_contract import validate_task1_output
from src.scripts.run_e028_official_only import ARCHITECTURES, read_partition, seed_all
from src.scripts.run_e029_generate import normalize_log1p_cp10k
from src.scripts.run_e029_grouped_official import generate_zero_weighted, group_state, state_loss
from src.scripts.run_m3_known_population import expression_metrics

EXTERNAL = ["E65_ex.h5ad","E675_ex.h5ad","E70_ex.h5ad","E725_ex.h5ad","E75_ex.h5ad",
            "E775_ex.h5ad","E80_ex.h5ad","E825_ex.h5ad","E85_ex.h5ad"]

def sample(path, maximum, seed):
    a=ad.read_h5ad(path, backed="r"); n=min(maximum,a.n_obs)
    rows=np.sort(np.random.default_rng(seed).choice(a.n_obs,n,replace=False)); x=a.X[rows]
    x=x.toarray() if sp.issparse(x) else np.asarray(x); genes=a.var_names.to_numpy(str); a.file.close()
    return x.astype(np.float32), genes

def fit(name, examples, val, dim, groups, args, device):
    arch=ARCHITECTURES[name]; torch.manual_seed(args.seed)
    model=GroupTransitionTransformer(dim,groups,arch["model_dim"],arch["heads"],arch["layers"],.05).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); ids=torch.arange(groups,device=device)[None]
    best=float("inf"); bad=0; checkpoint=args.output_dir/f"{name}_checkpoint.pt"; history=[]
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for features,target in examples:
            opt.zero_grad(); pred=model(torch.tensor(features[None],device=device),ids)
            tgt={k:torch.tensor(v,device=device) for k,v in target.items()}; loss=state_loss(pred,tgt,tgt["proportion"]>0)
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        model.eval(); features,target=val
        with torch.no_grad():
            tgt={k:torch.tensor(v,device=device) for k,v in target.items()}
            score=float(state_loss(model(torch.tensor(features[None],device=device),ids),tgt,tgt["proportion"]>0))
        history.append({"epoch":epoch,"train":float(np.mean(losses)),"validation":score})
        if score<best-1e-8:
            best=score; bad=0; torch.save({"state_dict":model.state_dict(),"epoch":epoch,"validation_loss":score},checkpoint)
        else:
            bad+=1
            if bad>=args.patience: break
    saved=torch.load(checkpoint,map_location=device,weights_only=False); model.load_state_dict(saved["state_dict"])
    (args.output_dir/f"{name}_history.json").write_text(json.dumps(history,indent=2)+"\n")
    return model,{"epoch":saved["epoch"],"validation_loss":saved["validation_loss"]}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data-dir",type=Path,default=Path("data")); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--latent-dim",type=int,default=32); p.add_argument("--groups",type=int,default=32); p.add_argument("--external-cells",type=int,default=512)
    p.add_argument("--train-cells",type=int,default=1024); p.add_argument("--validation-cells",type=int,default=256); p.add_argument("--source-pool-cells",type=int,default=2048); p.add_argument("--test-cells",type=int,default=512)
    p.add_argument("--input-genes",type=int,default=512); p.add_argument("--landmarks",type=int,default=256); p.add_argument("--gamma",type=float,default=.002); p.add_argument("--ridge",type=float,default=.01)
    p.add_argument("--epochs",type=int,default=250); p.add_argument("--patience",type=int,default=25); p.add_argument("--seed",type=int,default=42); p.add_argument("--threads",type=int,default=4); p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--official-only",action="store_true")
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); seed_all(args.seed,args.threads)
    e85,genes=read_partition(args.data_dir/"E85.h5ad",(args.train_cells,args.validation_cells,args.source_pool_cells),args.seed)
    e95,g95=read_partition(args.data_dir/"E95.h5ad",(args.train_cells,args.validation_cells,args.test_cells),args.seed+1)
    blocks=[]
    for i,name in enumerate([] if args.official_only else EXTERNAL):
        x,g=sample(args.data_dir/name,args.external_cells,args.seed+100+i)
        if not np.array_equal(genes,g): raise ValueError("Gene order mismatch")
        blocks.append(x)
    if not np.array_equal(genes,g95): raise ValueError("Gene order mismatch")
    basis=np.concatenate(blocks+[e85[0][0],e95[0][0]]); var=basis.var(0); selected=np.sort(np.argpartition(var,-args.input_genes)[-args.input_genes:])
    mean=basis[:,selected].mean(0); scale=basis[:,selected].std(0)+1e-4; norm=lambda x:((x[:,selected]-mean)/scale).astype(np.float32)
    enc=ApproximateKernelPCA(args.landmarks,args.latent_dim,args.gamma,args.ridge,args.seed); enc.fit_embedding(np.concatenate([norm(x) for x in blocks]+[norm(e85[0][0]),norm(e95[0][0])]))
    latent_blocks=[enc.transform(norm(x)) for x in blocks]; l85=[enc.transform(norm(x[0])) for x in e85]; l95=[enc.transform(norm(x[0])) for x in e95]
    enc.fit_decoder(latent_blocks+[l85[0],l95[0]],blocks+[e85[0][0],e95[0][0]])
    km=MiniBatchKMeans(args.groups,random_state=args.seed,n_init=5,batch_size=512).fit(np.concatenate(latent_blocks+[l85[0],l95[0]]))
    ext=[group_state(z,km) for z in latent_blocks]; s85=[group_state(z,km) for z in l85]; s95=[group_state(z,km) for z in l95]
    examples=[(ext[i][1],ext[i+1][0]) for i in range(len(ext)-1)]+[(s85[0][1],s95[0][0])]
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else args.device if args.device!="auto" else "cpu")
    predictions={"group_delta":{"proportion":s95[0][0]["proportion"].copy(),"latent_mean":s85[2][0]["latent_mean"]+s95[0][0]["latent_mean"]-s85[0][0]["latent_mean"],"latent_dispersion":np.maximum(s85[2][0]["latent_dispersion"]+s95[0][0]["latent_dispersion"]-s85[0][0]["latent_dispersion"],0)}}; checkpoints={}
    ids=torch.arange(args.groups,device=device)[None]
    for name in ("small","medium"):
        model,checkpoints[name]=fit(name,examples,(s85[1][1],s95[1][0]),args.latent_dim,args.groups,args,device); model.eval()
        with torch.no_grad(): raw=model(torch.tensor(s85[2][1][None],device=device),ids)
        predictions[name]={k:v[0].cpu().numpy().astype(np.float32) for k,v in raw.items()}
    support=np.zeros((args.groups,len(genes)),bool)
    for k in range(args.groups):
        members=e95[0][0][s95[0][2]==k]
        if len(members): support[k]=(members>0).any(0)
    metrics={}; contracts={}; target=e95[2][0]
    for name,pred in predictions.items():
        mats,labels=generate_zero_weighted(pred,s85[2][0],l85[2],e85[2][0],s85[2][2],enc,args.test_cells,args.seed,support,[.2]); matrix=normalize_log1p_cp10k(mats[.2])
        metrics[name]=expression_metrics(matrix,target,args.seed)|{"zero_fraction":float((matrix==0).mean()),"unique_cells":int(np.unique(matrix,axis=0).shape[0])}
        out=ad.AnnData(matrix,obs={"group_id":labels.astype(str)},var={"gene":genes}); out.var_names=genes; out.uns.update(experiment="E034" if args.official_only else "E033",model=name,external_ex_used=not args.official_only,seed=args.seed,cell_pairing=False)
        contracts[name]=validate_task1_output(out,genes,min_cells=args.test_cells,max_cells=args.test_cells); out.write_h5ad(args.output_dir/f"{name}.h5ad",compression="gzip")
    result={"metrics":metrics,"contracts":contracts,"checkpoints":checkpoints,"external_sources":[] if args.official_only else EXTERNAL,"latent_dim":args.latent_dim,"groups":args.groups}
    (args.output_dir/"metrics.json").write_text(json.dumps(result,indent=2,default=str)+"\n"); (args.output_dir/"config.json").write_text(json.dumps(vars(args),indent=2,default=str)+"\n"); print(json.dumps(result,indent=2,default=str))
if __name__=="__main__": main()
