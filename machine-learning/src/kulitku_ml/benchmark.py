import argparse
import json
import time
import subprocess
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix
from kulitku_core.images import LABELS,save_json,digest
from kulitku_ml.data import validate_frozen

def pipeline(key,seed):
    if key=="lr":return Pipeline([("scaler",StandardScaler()),("classifier",LogisticRegression(max_iter=3000,random_state=seed))])
    if key=="svm":return Pipeline([("scaler",StandardScaler()),("classifier",SVC(probability=False,random_state=seed))])
    if key=="rf":return Pipeline([("classifier",RandomForestClassifier(random_state=seed,n_jobs=1))])
    raise ValueError(key)

def folds(frame):
    indices=[]
    for k in sorted(frame.fold.unique()):
        train=np.flatnonzero((frame.fold!=k).to_numpy())
        val=np.flatnonzero(((frame.fold==k)&(frame.is_original==1)).to_numpy())
        if not len(train) or set(frame.iloc[val].target)!=set(range(4)):raise ValueError("Invalid grouped fold")
        if set(frame.iloc[train].group_id)&set(frame.iloc[val].group_id):raise ValueError("Group leakage")
        indices.append((train,val))
    return indices

def metrics(y,pred):
    return {"n":len(y),"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,labels=range(4),average="macro",zero_division=0)),
            "report":classification_report(y,pred,labels=range(4),target_names=LABELS,output_dict=True,zero_division=0),
            "confusion_matrix":confusion_matrix(y,pred,labels=range(4)).tolist(),
            "disease_predicted_normal":int(((np.asarray(y)!=2)&(pred==2)).sum())}

def run(features_root,config,out):
    cfg=json.loads(Path(config).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config_id=digest(config)
    if (out/"config.json").exists() and json.loads((out/"config.json").read_text())!=cfg:raise ValueError("Benchmark config changed")
    save_json(out/"config.json",cfg)
    if not (out/"environment.txt").exists():
        frozen=subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,check=True).stdout
        (out/"environment.txt").write_text(frozen,encoding="utf-8")
    records=[];common_manifest=None
    for key in cfg["models"]:
        source=Path(features_root)/key
        meta=json.loads((source/"metadata.json").read_text())
        progress=json.loads((source/"progress.json").read_text())
        if progress["completed"]!=progress["total"]:raise ValueError("Incomplete features")
        if meta["config_sha256"]!=config_id:raise ValueError("Extraction and benchmark configs differ")
        if common_manifest is None:common_manifest=meta["manifest_sha256"]
        elif common_manifest!=meta["manifest_sha256"]:raise ValueError("Different splits across CNNs")
        frame=validate_frozen(pd.read_csv(source/"manifest.csv",keep_default_na=False,dtype={"group_id":str,"sha256":str}))
        if digest(source/"manifest.csv")!=meta["manifest_sha256"]:raise ValueError("Feature manifest changed")
        features=np.load(source/"features.npy",mmap_mode="r")
        development=frame.split=="development"
        dev=frame[development].reset_index(drop=True)
        X=np.asarray(features[development]);y=dev.target.to_numpy();cv=folds(dev)
        for clf in ["lr","svm","rf"]:
            dest=out/f"{key}__{clf}";dest.mkdir(exist_ok=True)
            expected={"config_sha256":config_id,"feature_metadata_sha256":digest(source/"metadata.json")}
            if (dest/"summary.json").exists():
                saved=json.loads((dest/"summary.json").read_text())
                if saved["provenance"]!=expected:raise ValueError("Resume provenance changed")
                records.append(saved);continue
            search=GridSearchCV(pipeline(clf,cfg["seed"]),cfg["grids"][clf],scoring="f1_macro",cv=cv,
                                refit=True,n_jobs=cfg["n_jobs"],error_score="raise",return_train_score=True)
            begin=time.perf_counter();search.fit(X,y);seconds=time.perf_counter()-begin
            joblib.dump(search.best_estimator_,dest/"pipeline.joblib")
            pd.DataFrame(search.cv_results_).to_csv(dest/"cv_results.csv",index=False)
            row={"cnn":key,"classifier":clf,"cv_macro_f1_mean":float(search.best_score_),
                 "cv_macro_f1_std":float(search.cv_results_["std_test_score"][search.best_index_]),
                 "best_params":search.best_params_,"training_seconds":seconds,"provenance":expected,
                 "feature_dir":str(source.resolve()),"synthetic_demo":meta["synthetic_demo"]}
            save_json(dest/"summary.json",row);records.append(row);print(json.dumps(row),flush=True)
    table=pd.DataFrame(records).sort_values(["cv_macro_f1_mean","cv_macro_f1_std","cnn","classifier"],ascending=[False,True,True,True])
    table.to_csv(out/"leaderboard.csv",index=False)
    winner=table.iloc[0].to_dict()
    save_json(out/"winner.json",winner)
    print(table[["cnn","classifier","cv_macro_f1_mean","cv_macro_f1_std"]])


def evaluate(benchmark,out):
    benchmark=Path(benchmark);out=Path(out)
    if out.exists():raise ValueError("Final test output exists; do not overwrite")
    winner=json.loads((benchmark/"winner.json").read_text());source=Path(winner["feature_dir"])
    frame=validate_frozen(pd.read_csv(source/"manifest.csv",keep_default_na=False,dtype={"group_id":str,"sha256":str}))
    if digest(source/"metadata.json")!=winner["provenance"]["feature_metadata_sha256"]:raise ValueError("Feature provenance changed")
    meta=json.loads((source/"metadata.json").read_text())
    if digest(source/"manifest.csv")!=meta["manifest_sha256"]:raise ValueError("Manifest changed")
    mask=frame.split=="test";test=frame[mask].copy();X=np.load(source/"features.npy",mmap_mode="r")[mask]
    model=joblib.load(benchmark/f"{winner['cnn']}__{winner['classifier']}"/"pipeline.joblib") # Trust own local artifacts only.
    pred=model.predict(X);result=metrics(test.target.to_numpy(),pred)
    # Group-bootstrap CI resamples groups, not augmented files.
    rng=np.random.default_rng(42);groups=test.group_id.to_numpy();unique=np.unique(groups)
    positions={g:np.flatnonzero(groups==g) for g in unique};values=[];y=test.target.to_numpy()
    for _ in range(300):
        ix=np.concatenate([positions[g] for g in rng.choice(unique,len(unique),replace=True)])
        values.append(f1_score(y[ix],pred[ix],labels=range(4),average="macro",zero_division=0))
    result["macro_f1_group_bootstrap_95ci"]=np.quantile(values,[.025,.975]).tolist()
    result["synthetic_demo"]=winner["synthetic_demo"]
    out.mkdir(parents=True);save_json(out/"metrics.json",result)
    test["predicted_target"]=pred;test.to_csv(out/"predictions.csv",index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,ax=plt.subplots(figsize=(7,6));ConfusionMatrixDisplay(np.array(result["confusion_matrix"]),display_labels=LABELS).plot(ax=ax,xticks_rotation=30)
    fig.tight_layout();fig.savefig(out/"confusion_matrix.png");plt.close(fig)
    (out/"analysis.md").write_text(f"# Final evaluation draft\nWinner selected by development CV: {winner['cnn']} + {winner['classifier']}.\nMacro-F1: {result['macro_f1']:.4f}. Review per-class errors and false Normal cases in metrics.json/predictions.csv.\nNo independent smartphone or clinical validation is established by this internal test.\n",encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("run");a.add_argument("--features",required=True);a.add_argument("--config",required=True);a.add_argument("--output",required=True)
    a=sub.add_parser("evaluate");a.add_argument("--benchmark",required=True);a.add_argument("--output",required=True)
    a=p.parse_args()
    if a.cmd=="run":run(a.features,a.config,a.output)
    else:evaluate(a.benchmark,a.output)
