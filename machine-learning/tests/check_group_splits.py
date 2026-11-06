"""Check verified augmentation groups and stop on conflicting exact-duplicate labels."""
from pathlib import Path
import tempfile,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from PIL import Image
from kulitku_ml.config import load_config,Run
from kulitku_ml.data import inventory
from kulitku_ml.cleaning import clean
from kulitku_ml.splitting import split,cv_indices

root=Path(tempfile.mkdtemp(prefix='kulitku_verified_'));data=root/'data';rng=np.random.default_rng(42)
cfg=load_config(Path(__file__).resolve().parents[1]/'configs/experiment.json')
cfg['paths']={'dataset_root':str(data),'output_root':str(root/'hasil')}
cfg['split']['mode']='verified_groups';cfg['split']['lineage_csv']=str(root/'lineage.csv')
audit=[]
for label in cfg['classes']:
    (data/label).mkdir(parents=True)
    for i in range(16):
        a=rng.integers(0,255,(130,132,3),dtype=np.uint8)
        for version,arr in [('original',a),('augmented',a[:,::-1,:])]:
            name=f'{label}/{i}_{version}.png';Image.fromarray(arr).save(data/name)
            audit.append({'relative_path':name,'lineage_group':f'{label}_{i}',
                'original_status':'verified_'+version,'evidence':'Synthetic fixture with known generated parent'})
pd.DataFrame(audit).to_csv(root/'lineage.csv',index=False)
run=Run(cfg,'verified');df=split(run,clean(run,inventory(run)))
for tr,va in cv_indices(run,df):
    assert df.iloc[va].original_status.eq('verified_original').all()
    assert not set(df.iloc[tr].lineage_group)&set(df.iloc[va].lineage_group)
assert not set(df[df.partition.eq('test')].lineage_group)&set(df[df.partition.eq('development')].lineage_group)
assert df.groupby('lineage_group').partition.nunique().eq(1).all()
assert df[df.partition.eq('development')].groupby('lineage_group').fold.nunique().eq(1).all()
# Force cross-label byte duplicate; must raise with an exported conflict table.
import shutil
shutil.copy2(data/'Eczema/0_original.png',data/'Normal/conflict.png')
cfg['split']['mode']='exploratory_hash'
r2=Run(cfg,'conflict')
try:split(r2,clean(r2,inventory(r2)))
except ValueError as e:
    assert 'label berbeda' in str(e)
    assert (r2.out/'tabel/duplicate_label_conflicts.csv').exists()
else:raise AssertionError('Conflicting labels accepted')
print('PASS verified original-only validation, isolated augmentation groups, grouped holdout and CV, conflicting labels stop.')
