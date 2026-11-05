import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import timm
from timm.data import resolve_model_data_config
from kulitku_core.images import load_rgb,preprocess,save_json,digest
from kulitku_ml.data import validate_frozen

def extract(data,manifest,config,model_key,out,random_init=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(config).read_text());name=cfg["models"][model_key]
    frame=validate_frozen(pd.read_csv(manifest,keep_default_na=False,dtype={"group_id":str,"sha256":str}))
    signature={"manifest_sha256":digest(manifest),"config_sha256":digest(config),"model_key":model_key,"model_name":name,"random_init":random_init}
    progress=out/"progress.json"
    start=0
    if progress.exists():
        previous=json.loads(progress.read_text())
        if previous["signature"]!=signature:raise ValueError("Resume config/manifest/model mismatch")
        start=previous["completed"]
    random.seed(cfg["seed"]);np.random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"])
    torch.set_num_threads(min(torch.get_num_threads(),4))
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights=out/"backbone.pth"
    if start and not weights.exists():raise ValueError("Missing extraction weights")
    model=timm.create_model(name,pretrained=not random_init and not weights.exists(),num_classes=0,global_pool="avg")
    if weights.exists():model.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True))
    else:torch.save(model.state_dict(),weights)
    model.requires_grad_(False);model.eval();model.to(device)
    dc=resolve_model_data_config(model)
    spec={"size":cfg["image_size"],"mean":list(dc["mean"]),"std":list(dc["std"]),
          "interpolation":dc.get("interpolation","bicubic"),"resize":"stretch","layout":"NCHW","color":"RGB"}
    if spec["interpolation"] not in {"bilinear","bicubic","lanczos"}:raise ValueError("Unsupported interpolation")
    with torch.inference_mode():dim=int(model(torch.zeros(1,3,cfg["image_size"],cfg["image_size"],device=device)).shape[1])
    path=out/"features.npy"
    matrix=np.lib.format.open_memmap(path,mode="r+" if start else "w+",dtype=np.float32,shape=(len(frame),dim))
    for first in range(start,len(frame),cfg["batch_size"]):
        stop=min(first+cfg["batch_size"],len(frame));batch=[]
        for row in frame.iloc[first:stop].itertuples():
            source=Path(data)/row.path
            if digest(source)!=row.sha256:raise ValueError(f"Image changed after audit: {row.path}")
            batch.append(preprocess(load_rgb(source),spec)[0])
        with torch.inference_mode():values=model(torch.from_numpy(np.stack(batch)).to(device)).cpu().numpy()
        if values.shape!=(stop-first,dim) or not np.isfinite(values).all():raise ValueError("Invalid features")
        matrix[first:stop]=values;matrix.flush()
        save_json(progress,{"signature":signature,"completed":stop,"total":len(frame)})
        print(f"{model_key}: {stop}/{len(frame)}",flush=True)
    frame.to_csv(out/"manifest.csv",index=False)
    save_json(out/"preprocessing.json",spec)
    save_json(out/"metadata.json",{**signature,"feature_dim":dim,"rows":len(frame),"weights_sha256":digest(weights),
        "torch":torch.__version__,"timm":timm.__version__,"synthetic_demo":random_init})
    print("Features complete:",out)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ["data","manifest","config","model","output"]:p.add_argument("--"+key,required=True)
    p.add_argument("--random-init",action="store_true");a=p.parse_args()
    extract(a.data,a.manifest,a.config,a.model,a.output,a.random_init)
