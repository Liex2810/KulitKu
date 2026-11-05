import json
from pathlib import Path
import joblib
import numpy as np
import onnxruntime as ort
from kulitku_core.images import LABELS,preprocess,digest
class Engine:
    def __init__(self,settings):
        self.demo=settings.demo_mode
        if self.demo:
            self.release={"version":"technical-demo","synthetic_demo":True};return
        root=Path(settings.model_dir)
        checks=json.loads((root/"checksums.json").read_text())
        required={"extractor.onnx","classifier.joblib","preprocessing.json","labels.json","release.json"}
        if not required.issubset(checks):raise ValueError("Incomplete model bundle")
        for name,value in checks.items():
            if Path(name).name!=name or digest(root/name)!=value:raise ValueError("Bundle checksum mismatch")
        self.release=json.loads((root/"release.json").read_text());self.demo=self.release.get("synthetic_demo",False)
        import sklearn
        if self.release.get("sklearn_version")!=sklearn.__version__:
            raise ValueError(f"Install scikit-learn=={self.release.get('sklearn_version')} to match training")
        self.spec=json.loads((root/"preprocessing.json").read_text())
        if json.loads((root/"labels.json").read_text())!=LABELS:raise ValueError("Unexpected class order")
        # joblib is pickle based: deploy only bundles produced by your own trusted pipeline.
        self.classifier=joblib.load(root/"classifier.joblib")
        if list(self.classifier.classes_)!=list(range(4)):raise ValueError("Classifier classes do not match labels")
        self.session=ort.InferenceSession(str(root/"extractor.onnx"),providers=["CPUExecutionProvider"])
    def predict(self,image):
        if self.demo:return {"status":"demo_only","predicted_class":None,"probabilities":None,"quality_flags":[],"model_version":self.release["version"],"synthetic_demo":True,"message":"Demo koneksi, tidak menilai kondisi kulit."}
        features=self.session.run(None,{"image":preprocess(image,self.spec)})[0]
        if features.shape!=(1,self.release["feature_dim"]) or not np.isfinite(features).all():raise ValueError("Invalid feature vector")
        index=int(self.classifier.predict(features)[0]);probs=None
        if hasattr(self.classifier,"predict_proba"):
            values=self.classifier.predict_proba(features)[0]
            probs=dict(zip(LABELS,map(float,values)))
        return {"status":"classified","predicted_class":LABELS[index],"probabilities":probs,"quality_flags":[],
            "model_version":self.release["version"],"synthetic_demo":False,
            "message":"Hasil skrining model, bukan diagnosis. Skor belum dikalibrasi; kondisi di luar empat kelas dapat salah diklasifikasikan."}
