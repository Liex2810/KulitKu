"""Inventory -> manual augmentation audit -> reviewed group split. No guessed lineage."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import imagehash
from sklearn.model_selection import StratifiedGroupKFold
from kulitku_core.images import LABELS, load_rgb, digest, save_json

COLUMNS=["path","label","target","sha256","phash","width","height","source","group_id","is_original","include"]

def inventory(data,out):
    data=Path(data).resolve();out=Path(out)
    if out.exists():raise ValueError("Choose a new inventory file; do not overwrite reviewed metadata")
    rows=[];bad=[]
    aliases={"Eczema":["Eczema","ECZEMA","Eczema Photos"],"Herpes Zoster":["Herpes Zoster","HERPES ZOSTER","Vi Shingles"],
             "Normal":["Normal","NORMAL","Heathy"],"Ringworm":["Ringworm","RINGWORM","Fu Ringworm"]}
    for k,label in enumerate(LABELS):
        folders=[data/name for name in aliases[label] if (data/name).is_dir()]
        if len(folders)!=1:raise ValueError(f"Expected exactly one folder for {label}: {folders}")
        for path in sorted(folders[0].rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg",".jpeg",".png",".webp"}:continue
            try:
                im=load_rgb(path)
                rows.append({"path":path.relative_to(data).as_posix(),"label":label,"target":k,"sha256":digest(path),
                    "phash":str(imagehash.phash(im)),"width":im.width,"height":im.height,
                    "source":"kaggle-massive-skin-v1","group_id":"","is_original":"unknown","include":1})
            except ValueError as e:bad.append({"path":str(path),"error":str(e)})
    if not rows:raise ValueError("No images")
    out.parent.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows,columns=COLUMNS);frame.to_csv(out,index=False)
    frame[frame.duplicated("sha256",keep=False)].to_csv(out.with_name("exact_duplicates.csv"),index=False)
    frame[frame.duplicated("phash",keep=False)].to_csv(out.with_name("phash_candidates.csv"),index=False)
    save_json(out.with_suffix(".errors.json"),bad)
    eda(frame,out.parent/"eda_inventory")
    print("Review group_id / is_original / include in a COPY of inventory before splitting.")

def eda(frame,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    frame.groupby("label").size().to_csv(out/"images_per_class.csv")
    pd.crosstab(frame.label,frame.source).to_csv(out/"class_by_source.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    frame.label.value_counts().plot.bar(ax=ax[0]);ax[0].set_title("Images per class")
    ax[1].hist(frame.width,bins=25);ax[1].set_title("Width")
    ax[2].hist(frame.height,bins=25);ax[2].set_title("Height")
    fig.tight_layout();fig.savefig(out/"eda.png");plt.close(fig)

def checked(frame):
    required={"path","label","target","sha256","group_id","is_original","include"}
    if not required.issubset(frame):raise ValueError(f"Missing {required-set(frame.columns)}")
    frame=frame.copy()
    frame["include"]=pd.to_numeric(frame["include"],errors="raise")
    if not frame.include.isin([0,1]).all():raise ValueError("include must be 0 or 1")
    frame=frame[frame.include==1].copy().reset_index(drop=True)
    if frame.empty or frame.path.duplicated().any():raise ValueError("Empty dataset or duplicate paths")
    if frame.group_id.astype(str).str.strip().eq("").any():raise ValueError("Complete group_id before splitting")
    if not frame.is_original.astype(str).isin(["0","1"]).all():raise ValueError("is_original must be reviewed as 0 or 1; unknown is not accepted")
    frame["is_original"]=frame.is_original.astype(int)
    for r in frame.itertuples():
        if int(r.target) not in range(4) or LABELS[int(r.target)]!=r.label:raise ValueError("Label mismatch")
        if Path(r.path).is_absolute() or ".." in Path(r.path).parts:raise ValueError("Unsafe relative path")
    if frame.groupby("sha256").group_id.nunique().max()>1:raise ValueError("Exact duplicates have different group_id values")
    if frame.groupby("sha256").target.nunique().max()>1:raise ValueError("Exact duplicate has conflicting labels")
    # Prefer a reviewed original if an exact duplicate is marked augmented.
    frame=frame.sort_values("is_original",ascending=False,kind="stable").drop_duplicates("sha256").reset_index(drop=True)
    return frame

def split(manifest,out,config,approved=False):
    if not approved:raise ValueError("Manual audit required: review metadata, then pass --approve-audit")
    out=Path(out)
    if out.exists():raise ValueError("Split directory already exists; choose a new version")
    cfg=json.loads(Path(config).read_text());frame=checked(pd.read_csv(manifest,keep_default_na=False,dtype={"group_id":str,"sha256":str}))
    # All groups need a reviewed original, so held-out evaluation has independent originals.
    if frame.groupby("group_id").is_original.sum().eq(0).any():raise ValueError("Each group needs a reviewed original; obtain original data if unavailable")
    if frame[frame.is_original==1].groupby("target").group_id.nunique().reindex(range(4),fill_value=0).min()<cfg["test_folds"]:
        raise ValueError("Too few independent groups per class")
    outer=StratifiedGroupKFold(cfg["test_folds"],shuffle=True,random_state=cfg["seed"])
    _,hold=next(outer.split(frame,frame.target,frame.group_id))
    frame["split"]="development";frame.loc[hold,"split"]="test"
    dev=frame[frame.split=="development"].copy()
    if dev[dev.is_original==1].groupby("target").group_id.nunique().reindex(range(4),fill_value=0).min()<cfg["folds"]:
        raise ValueError("Too few development groups for requested folds")
    frame["fold"]=-1
    inner=StratifiedGroupKFold(cfg["folds"],shuffle=True,random_state=cfg["seed"])
    for fold,(_,val) in enumerate(inner.split(dev,dev.target,dev.group_id)):
        frame.loc[dev.index[val],"fold"]=fold
    test=frame[(frame.split=="test")&(frame.is_original==1)]
    if set(test.target)!=set(range(4)):raise ValueError("Test originals missing classes")
    for fold in range(cfg["folds"]):
        if set(frame[(frame.split=="development")&(frame.fold==fold)&(frame.is_original==1)].target)!=set(range(4)):
            raise ValueError("Validation originals missing classes; reduce folds or collect data")
    # Quarantine augmented test derivatives: never extract/train/evaluate them.
    frame=frame[~((frame.split=="test")&(frame.is_original==0))].reset_index(drop=True)
    out.mkdir(parents=True);frame.to_csv(out/"manifest.csv",index=False)
    save_json(out/"provenance.json",{"reviewed_manifest_sha256":digest(manifest),"seed":cfg["seed"],"folds":cfg["folds"],"manual_audit_approved":True})
    eda(frame,out/"eda_cleaned")
    frame.groupby(["split","label"]).agg(images=("path","size"),groups=("group_id","nunique")).to_csv(out/"counts.csv")
    print(frame.groupby(["split","label"]).size())

def validate_frozen(frame):
    frame=checked(frame)
    if not {"split","fold"}.issubset(frame):raise ValueError("Not a frozen manifest")
    if not frame.split.isin(["development","test"]).all():raise ValueError("Unknown split")
    if frame.groupby("group_id").split.nunique().max()>1:raise ValueError("Group leakage across splits")
    dev=frame[frame.split=="development"]
    if dev.groupby("group_id").fold.nunique().max()>1:raise ValueError("Group leakage across folds")
    if (frame[frame.split=="test"].is_original!=1).any():raise ValueError("Test must contain originals only")
    return frame

if __name__=="__main__":
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("inventory");a.add_argument("--data",required=True);a.add_argument("--output",required=True)
    a=sub.add_parser("split");a.add_argument("--manifest",required=True);a.add_argument("--output",required=True);a.add_argument("--config",required=True);a.add_argument("--approve-audit",action="store_true")
    a=p.parse_args()
    if a.cmd=="inventory":inventory(a.data,a.output)
    else:split(a.manifest,a.output,a.config,a.approve_audit)
