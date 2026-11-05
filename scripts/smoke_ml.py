"""Technical end-to-end smoke using synthetic images and random CNN weights.
No network downloads, no medical-performance claims. Outputs outside repository.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from kulitku_core.images import LABELS

def main(output):
    base=Path(output)
    if base.exists():raise ValueError('Use a fresh smoke directory')
    base.mkdir(parents=True)
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'machine-learning/configs/default.json').read_text())
    config.update(folds=2,test_folds=3,batch_size=4,n_jobs=1,num_workers=0)
    config['grids']={'lr':{'classifier__C':[1.0]},'svm':{'classifier__C':[1.0],'classifier__kernel':['linear']},'rf':{'classifier__n_estimators':[5],'classifier__max_depth':[3]}}
    cfg=base/'config.json';cfg.write_text(json.dumps(config))
    metadata=[]
    for k,label in enumerate(LABELS):
        folder=base/'data'/label;folder.mkdir(parents=True)
        for group in range(7):
            pixels=np.random.default_rng(100*k+group).integers(20,235,(128,128,3),dtype=np.uint8)
            for original in [1,0]:
                name=f'{group}-{original}.png'
                Image.fromarray(pixels if original else np.flip(pixels,axis=1)).save(folder/name)
                metadata.append({'path':f'{label}/{name}','group_id':f'synthetic-patient-{k}-{group}','is_original':original})
    def run(module,*args):subprocess.run([sys.executable,'-m',module,*map(str,args)],cwd=root,check=True)
    run('kulitku_ml.data','inventory','--data',base/'data','--output',base/'inventory.csv')
    frame=pd.read_csv(base/'inventory.csv');frame=frame.drop(columns=['group_id','is_original']).merge(pd.DataFrame(metadata),on='path',validate='one_to_one')
    reviewed=base/'reviewed.csv';frame.to_csv(reviewed,index=False)
    run('kulitku_ml.data','split','--manifest',reviewed,'--config',cfg,'--output',base/'split','--approve-audit')
    for key in config['models']:
        run('kulitku_ml.features','--data',base/'data','--manifest',base/'split/manifest.csv','--config',cfg,'--model',key,'--output',base/'features'/key,'--random-init')
    run('kulitku_ml.benchmark','run','--features',base/'features','--config',cfg,'--output',base/'benchmark')
    # Resume completed combinations must preserve provenance and skip completed searches.
    run('kulitku_ml.benchmark','run','--features',base/'features','--config',cfg,'--output',base/'benchmark')
    run('kulitku_ml.benchmark','evaluate','--benchmark',base/'benchmark','--output',base/'evaluation')
    run('kulitku_ml.export','--benchmark',base/'benchmark','--output',base/'bundle','--version','synthetic-smoke','--data',base/'data')
    from app.engine import Engine
    from app.settings import Settings
    from kulitku_core.images import load_rgb
    engine=Engine(Settings(auth_mode='local',persist_history=False,model_dir=str(base/'bundle')))
    result=engine.predict(load_rgb(base/'data/Eczema/0-1.png'))
    assert result['status']=='demo_only' and result['predicted_class'] is None
    assert len(pd.read_csv(base/'benchmark/leaderboard.csv'))==9
    print('PASS: 3 random CNNs, 9 classifier combinations, grouped CV, resume, evaluation, ONNX export and API engine. SYNTHETIC ONLY.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
