"""Actual consumer of exported labels/preprocessing; usable by a future FastAPI endpoint."""
import json,time
from pathlib import Path
import numpy as np
from .preprocessing import rgb,build_transform


class Predictor:
    def __init__(self,folder,threads=2):
        import onnxruntime as ort
        self.folder=Path(folder)
        release=json.loads((self.folder/'release.json').read_text())
        if not release['parity_passed']:raise ValueError('Release belum diverifikasi.')
        import hashlib
        for name,expected in release['checksums'].items():
            if hashlib.sha256((self.folder/name).read_bytes()).hexdigest()!=expected:raise ValueError('Checksum release tidak cocok.')
        self.labels=json.loads((self.folder/'labels.json').read_text())['classes']
        self.spec=json.loads((self.folder/'preprocessing.json').read_text())
        self.transform=build_transform(self.spec['transform'])
        opts=ort.SessionOptions();opts.intra_op_num_threads=threads;opts.inter_op_num_threads=1
        self.extractor=ort.InferenceSession(str(self.folder/'extractor.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
        self.classifier=ort.InferenceSession(str(self.folder/'classifier.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])

    def predict(self,path):
        start=time.perf_counter()
        image=rgb(path,self.spec['max_pixels'])
        tensor=self.transform(image).unsqueeze(0).numpy().astype('float32')
        feature=self.extractor.run(None,{self.extractor.get_inputs()[0].name:tensor})[0]
        label=self.classifier.run([self.classifier.get_outputs()[0].name],{self.classifier.get_inputs()[0].name:feature})[0]
        index=int(np.asarray(label).reshape(-1)[0])
        return {'label':self.labels[index],'class_index':index,'milliseconds':(time.perf_counter()-start)*1000}
