"""Compare full CPU pipeline latency on development originals, never test selection."""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch,timm
from kulitku_core.images import load_rgb,preprocess

def profile(benchmark,data,samples=20):
    benchmark=Path(benchmark);table=pd.read_csv(benchmark/"leaderboard.csv");rows=[]
    torch.set_num_threads(2)
    for cnn,subset in table.groupby("cnn"):
        source=Path(subset.iloc[0].feature_dir);meta=json.loads((source/"metadata.json").read_text());spec=json.loads((source/"preprocessing.json").read_text())
        model=timm.create_model(meta["model_name"],pretrained=False,num_classes=0,global_pool="avg")
        model.load_state_dict(torch.load(source/"backbone.pth",map_location="cpu",weights_only=True));model.eval()
        manifest=pd.read_csv(source/"manifest.csv");paths=manifest[(manifest.split=="development")&(manifest.is_original==1)].path.head(samples)
        for item in subset.itertuples():
            clf=joblib.load(benchmark/f"{cnn}__{item.classifier}"/"pipeline.joblib");times=[]
            for repetition in range(2):
                for path in paths:
                    image=load_rgb(Path(data)/path) # Excludes disk/image decode; consistent for all candidates.
                    t=time.perf_counter();x=preprocess(image,spec)
                    with torch.inference_mode():features=model(torch.from_numpy(x)).numpy()
                    clf.predict(features)
                    if repetition:times.append((time.perf_counter()-t)*1000)
            rows.append({"cnn":cnn,"classifier":item.classifier,"pipeline_cpu_p50_ms":float(np.median(times)),"pipeline_cpu_p95_ms":float(np.percentile(times,95)),"extractor_state_bytes":(source/"backbone.pth").stat().st_size,"classifier_bytes":(benchmark/f"{cnn}__{item.classifier}"/"pipeline.joblib").stat().st_size,"samples":len(times),"threads":2,"runtime":"PyTorch+sklearn","includes":"preprocessing+CNN+scaler+classifier"})
    pd.DataFrame(rows).to_csv(benchmark/"latency_comparison.csv",index=False)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--benchmark",required=True);p.add_argument("--data",required=True);p.add_argument("--samples",type=int,default=20)
    a=p.parse_args();profile(a.benchmark,a.data,a.samples)
