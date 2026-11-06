"""Synthetic integration test. Random CNN weights, tiny grids; NOT medical performance."""
import os,sys,tempfile,json
from pathlib import Path
os.environ['MPLBACKEND']='Agg';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
plt.show=lambda:None
from PIL import Image
import torch
from kulitku_ml.config import load_config,Run
from kulitku_ml.data import inventory
from kulitku_ml.cleaning import clean
from kulitku_ml.eda import eda,compare
from kulitku_ml.splitting import split,cv_indices
from kulitku_ml.features import extract_features
from kulitku_ml.training import train_classifier
from kulitku_ml.evaluation import validation_report,cv_overview,final_test
from kulitku_ml.benchmarking import benchmark
from kulitku_ml.export import export_selected
from kulitku_ml.inference import Predictor

def main():
    torch.set_num_threads(2)
    root=Path(tempfile.mkdtemp(prefix='kulitku_complete_smoke_'));data=root/'dataset'
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/experiment.json')
    cfg['paths']={'dataset_root':str(data),'output_root':str(root/'hasil')}
    cfg['eda']={'sample_per_class':2,'gallery_per_class':2}
    cfg['benchmark']={'threads':2,'samples_per_class':1,'warmup':1,'repeats':1}
    cfg['evaluation']['bootstrap_repeats']=30;cfg['export']['parity_per_class']=1
    cfg['features']['batch_size']=8
    # Actual target architectures exercised, but random weights avoid external downloads.
    for spec in cfg['extractors'].values():spec['pretrained']=False
    cfg['classifiers']={
      'logistic_regression':{'model__C':[1.]},
      'svm':{'model__C':[1.],'model__kernel':['linear']},
      'random_forest':{'model__n_estimators':[12],'model__max_depth':[6]}}
    rng=np.random.default_rng(42)
    for ci,label in enumerate(cfg['classes']):
        (data/label).mkdir(parents=True)
        for i in range(16):
            arr=rng.integers(0,30,(144,160,3),dtype=np.uint8)
            arr[:,:,ci%3]=150+ci*20
            if ci==3:arr[:,::3,:]=220
            Image.fromarray(arr).save(data/label/f'{i:02d}.png')
    (data/'Eczema'/'broken.jpg').write_bytes(b'broken')
    (data/'Normal'/'notes.txt').write_text('sidecar')
    run=Run(cfg,'smoke')
    before=inventory(run);assert len(before)==66
    eda(run,before,'before')
    after=clean(run,before);assert len(after)==64
    compare(run,before,after);eda(run,after,'after')
    df=split(run,after)
    for tr,va in cv_indices(run,df):
        assert not set(df.iloc[tr].split_group)&set(df.iloc[va].split_group)
        assert df.iloc[tr].partition.eq('development').all()
        assert df.iloc[va].partition.eq('development').all()
    for cnn in cfg['extractors']:
        print('EXTRACT',cnn,flush=True);x=extract_features(run,df,cnn)
        assert np.isfinite(x).all()
        cached=extract_features(run,df,cnn);assert np.array_equal(x,cached)
        for name in cfg['classifiers']:
            result=train_classifier(run,df,x,cnn,name)
            assert len(run.read(f'cv_{cnn}_{name}.csv'))==5
            assert train_classifier(run,df,x,cnn,name)==result
            validation_report(run,cnn,name)
    table=cv_overview(run);assert len(table)==9
    bench=benchmark(run,df);assert len(bench)==9
    selected=json.loads((run.out/'selection.json').read_text())
    assert selected['mean_macro_f1']>=bench.mean_macro_f1.max()-.01
    final_test(run,df,selected['cnn'],selected['classifier'])
    # Verify ONNX CNN for ALL THREE target architectures, plus classifier pipelines.
    for cnn in cfg['extractors']:
        choice=dict(selected);choice['cnn']=cnn;choice['classifier']='logistic_regression';choice['combo_id']=cnn+'__logistic_regression'
        run.json(choice,'selection.json')
        folder=export_selected(run,df)
        predictor=Predictor(folder,threads=2)
        out=predictor.predict(data/df.iloc[0].relative_path)
        assert out['label'] in cfg['classes']
    # All 9 fitted sklearn pipelines convert; compare prediction labels on development features.
    import joblib,onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    for cnn in cfg['extractors']:
        x=np.load(run.out/'features'/cnn/'features.npy')[:8].astype('float32')
        for name in cfg['classifiers']:
            pipe=joblib.load(run.out/'models'/cnn/name/'classifier.joblib')
            onx=convert_sklearn(pipe,initial_types=[('features',FloatTensorType([None,x.shape[1]]))],options={id(pipe.named_steps['model']):{'zipmap':False}},target_opset=17)
            opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
            sess=ort.InferenceSession(onx.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider'])
            pred=sess.run([sess.get_outputs()[0].name],{'features':x})[0]
            assert np.array_equal(pipe.predict(x),np.asarray(pred).reshape(-1)),(cnn,name)
    run.json(selected,'selection.json')
    # Config/cache identity prevents accidental protocol change in the same run.
    cfg['seed']=99
    try:Run(cfg,'smoke')
    except ValueError:pass
    else:raise AssertionError('Changed config was accepted into prior run.')
    print('PASS full synthetic pipeline, 3 actual CNNs, 9 classifiers, 5 folds, ONNX parity, cache, locked config.')
    print('OUTPUT',run.out)
    return run.out

if __name__=='__main__':main()
