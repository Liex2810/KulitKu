import copy, hashlib, importlib.metadata as md, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def load_config(path):
    c=json.loads(Path(path).read_text())
    if c['classes']!=['Eczema','Herpes Zoster','Normal','Ringworm']:
        raise ValueError('Urutan empat kelas harus tetap.')
    if c['split']['folds']<2 or c['split']['holdout_folds']<2:
        raise ValueError('Jumlah fold minimal 2.')
    if c['split']['mode'] not in ['exploratory_hash','verified_groups']:
        raise ValueError('Mode split tidak dikenal.')
    if c['cleaning']['exclude_below_minimum'] and not c['cleaning']['minimum_reason']:
        raise ValueError('Alasan aturan minimum wajib dicatat.')
    if not 0<=c['selection']['macro_f1_tolerance']<1:raise ValueError('Toleransi tidak valid.')
    return c


def _json_default(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,Path):return str(v)
    raise TypeError(type(v).__name__)


class Run:
    def __init__(self,c,run_id=None):
        self.c=copy.deepcopy(c)
        self.root=Path(c['paths']['dataset_root'])
        self.id=run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]+',self.id):raise ValueError('run_id tidak aman.')
        self.out=Path(c['paths']['output_root'])/self.id
        for name in ['grafik','tabel','features','models','onnx']: (self.out/name).mkdir(parents=True,exist_ok=True)
        versions={}
        for name in ['numpy','pandas','Pillow','scikit-learn','torch','torchvision','timm','onnx','onnxruntime','skl2onnx']:
            try:versions[name]=md.version(name)
            except md.PackageNotFoundError:pass
        source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
        identity={'config':self.c,'source':source,'python':platform.python_version(),'packages':versions}
        self.fingerprint=digest(identity)
        lock=self.out/'run_config.json'
        if lock.exists():
            if json.loads(lock.read_text())['fingerprint']!=self.fingerprint:
                raise ValueError('Config/kode/versi berubah: gunakan RUN_ID baru. Jangan campur hasil lama.')
        else:self.json({'fingerprint':self.fingerprint,**identity},'run_config.json')
        self.json({'python':sys.version,'platform':platform.platform(),'packages':versions},'environment.json')
        self.exploratory=c['split']['mode']=='exploratory_hash'
        self.tag='[EKSPLORATORI; lineage belum diverifikasi] ' if self.exploratory else ''

    def json(self,value,name):
        path=self.out/name;path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.name+'.partial')
        temp.write_text(json.dumps(value,indent=2,default=_json_default))
        temp.replace(path);return path

    def table(self,frame,name):
        path=self.out/'tabel'/name;path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.name+'.partial');frame.to_csv(temp,index=False);temp.replace(path)
        return path

    def figure(self,fig,name,show=True):
        import matplotlib.pyplot as plt
        path=self.out/'grafik'/name;path.parent.mkdir(parents=True,exist_ok=True)
        fig.tight_layout();fig.savefig(path,dpi=150,bbox_inches='tight')
        if show:plt.show()
        plt.close(fig);return path

    def read(self,name):
        return pd.read_csv(self.out/'tabel'/name,keep_default_na=False)
