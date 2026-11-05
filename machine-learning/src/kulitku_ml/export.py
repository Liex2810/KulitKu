import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import timm
import torch
import onnx
import onnxruntime as ort
from kulitku_core.images import LABELS,save_json,digest,load_rgb,preprocess

def export(benchmark,output,version,data):
    benchmark=Path(benchmark);destination=Path(output)
    if destination.exists():raise ValueError("Export path already exists")
    winner=json.loads((benchmark/"winner.json").read_text());source=Path(winner["feature_dir"])
    if digest(source/"metadata.json")!=winner["provenance"]["feature_metadata_sha256"]:raise ValueError("Provenance changed")
    meta=json.loads((source/"metadata.json").read_text());spec=json.loads((source/"preprocessing.json").read_text())
    if digest(source/"backbone.pth")!=meta["weights_sha256"]:raise ValueError("CNN weights changed")
    model=timm.create_model(meta["model_name"],pretrained=False,num_classes=0,global_pool="avg")
    model.load_state_dict(torch.load(source/"backbone.pth",map_location="cpu",weights_only=True));model.eval()
    torch.set_num_threads(min(torch.get_num_threads(),4))
    clf_path=benchmark/f"{winner['cnn']}__{winner['classifier']}"/"pipeline.joblib"
    clf=joblib.load(clf_path)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent,prefix=".export-") as tmp:
        out=Path(tmp)
        sample=torch.zeros(1,3,spec["size"],spec["size"])
        torch.onnx.export(model,sample,str(out/"extractor.onnx"),input_names=["image"],output_names=["features"],opset_version=18,dynamo=True,external_data=False)
        onnx.checker.check_model(onnx.load(out/"extractor.onnx"))
        session=ort.InferenceSession(str(out/"extractor.onnx"),providers=["CPUExecutionProvider"])
        rng=np.random.default_rng(42);diffs=[];latencies=[]
        # Numerical conversion checks use synthetic tensors, NOT clinical performance claims.
        for _ in range(5):
            x=rng.normal(size=sample.shape).astype(np.float32)
            with torch.inference_mode():expected=model(torch.from_numpy(x)).numpy()
            t=time.perf_counter();actual=session.run(None,{"image":x})[0];latencies.append((time.perf_counter()-t)*1000)
            np.testing.assert_allclose(expected,actual,rtol=1e-3,atol=1e-4)
            if not np.array_equal(clf.predict(expected),clf.predict(actual)):raise ValueError("Classifier disagreement after ONNX conversion")
            diffs.append(float(np.max(np.abs(expected-actual))))
        manifest=pd.read_csv(source/"manifest.csv")
        if digest(source/"manifest.csv")!=meta["manifest_sha256"]:raise ValueError("Manifest changed")
        reference=np.load(source/"features.npy",mmap_mode="r")
        selected=manifest[(manifest.split=="development")&(manifest.is_original==1)].groupby("target",group_keys=False).head(2)
        for index,row in selected.iterrows():
            image=load_rgb(Path(data)/row.path);x=preprocess(image,spec)
            actual=session.run(None,{"image":x})[0];expected=np.asarray(reference[index:index+1])
            np.testing.assert_allclose(expected,actual,rtol=1e-3,atol=1e-4)
            if not np.array_equal(clf.predict(expected),clf.predict(actual)):raise ValueError("Real validation class disagreement")
            diffs.append(float(np.max(np.abs(expected-actual))))
        shutil.copy2(clf_path,out/"classifier.joblib");save_json(out/"preprocessing.json",spec);save_json(out/"labels.json",LABELS)
        save_json(out/"release.json",{"version":version,"cnn":winner["cnn"],"model_name":meta["model_name"],"classifier":winner["classifier"],
            "feature_dim":meta["feature_dim"],"synthetic_demo":winner["synthetic_demo"],"format":"onnx-extractor+sklearn-pipeline",
            "sklearn_version":__import__('sklearn').__version__,"manifest_sha256":meta["manifest_sha256"]})
        save_json(out/"parity.json",{"max_abs_difference":max(diffs),"synthetic_tensors":5,"validation_images":len(selected),"extractor_cpu_latency_ms_p50":float(np.median(latencies)),
            "note":"Numerical parity only; not clinical validation"})
        (out/"model_card.md").write_text("# KulitKu research bundle\nNot a medical diagnosis. Four classes only; no OOD detector or clinical validation. Load joblib only from a trusted source.\n",encoding="utf-8")
        save_json(out/"checksums.json",{p.name:digest(p) for p in out.iterdir() if p.is_file()})
        out.chmod(0o755);shutil.move(str(out),str(destination))
    print("Bundle exported:",destination)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--benchmark",required=True);p.add_argument("--output",required=True);p.add_argument("--version",required=True);p.add_argument("--data",required=True)
    a=p.parse_args();export(a.benchmark,a.output,a.version,a.data)
