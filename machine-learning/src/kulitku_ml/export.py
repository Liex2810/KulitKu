import json,time,hashlib
import numpy as np
import pandas as pd
import joblib
from .features import restore_extractor
from .preprocessing import build_transform,rgb


def export_selected(run,df):
    import torch,onnx,onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    if not (run.out/'test_metrics.json').exists():raise ValueError('Evaluasi test final dahulu.')
    selected=json.loads((run.out/'selection.json').read_text());cnn=selected['cnn'];name=selected['classifier']
    folder=run.out/'onnx'/selected['combo_id'];folder.mkdir(parents=True,exist_ok=True)
    # Do not leave a prior ready marker when a new conversion/parity fails.
    (folder/'release.json').unlink(missing_ok=True)
    model,info=restore_extractor(run,cnn);transform=build_transform(info['preprocessing'])
    pipeline=joblib.load(run.out/'models'/cnn/name/'classifier.joblib')
    dummy=torch.zeros(1,*info['preprocessing']['input_size'])
    torch.onnx.export(model,dummy,str(folder/'extractor.onnx'),input_names=['images'],output_names=['features'],
        dynamic_axes={'images':{0:'batch'},'features':{0:'batch'}},opset_version=17,dynamo=False)
    converted=convert_sklearn(pipeline,initial_types=[('features',FloatTensorType([None,info['shape'][1]]))],
        options={id(pipeline.named_steps['model']):{'zipmap':False}},target_opset=17)
    (folder/'classifier.onnx').write_bytes(converted.SerializeToString())
    onnx.checker.check_model(str(folder/'extractor.onnx'));onnx.checker.check_model(converted)
    opts=ort.SessionOptions();opts.intra_op_num_threads=run.c['benchmark']['threads'];opts.inter_op_num_threads=1
    ext=ort.InferenceSession(str(folder/'extractor.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    clf=ort.InferenceSession(str(folder/'classifier.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    eligible=df.evaluation_eligible.astype(str).str.lower().eq('true')
    pool=df[df.partition.eq('development')&eligible]
    samples=pd.concat([pool[pool.label.eq(c)].head(run.c['export']['parity_per_class']) for c in run.c['classes']])
    checks=[]
    for r in samples.itertuples():
        batch=transform(rgb(run.root/r.relative_path)).unsqueeze(0)
        with torch.inference_mode():native=model(batch).numpy().astype('float32')
        exported=ext.run(None,{'images':batch.numpy()})[0]
        expected=pipeline.predict(native)
        actual=clf.run([clf.get_outputs()[0].name],{clf.get_inputs()[0].name:exported})[0]
        difference=float(np.max(np.abs(native-exported)))
        ok=bool(np.allclose(native,exported,atol=1e-4,rtol=1e-3)) and bool(np.array_equal(expected,np.asarray(actual).reshape(-1)))
        checks.append({'image_id':r.image_id,'max_feature_error':difference,'label_match':bool(np.array_equal(expected,np.asarray(actual).reshape(-1))),'passed':ok})
    table=pd.DataFrame(checks);run.table(table,'onnx_parity.csv')
    if table.empty or not table.passed.all():raise ValueError('ONNX parity gagal. Tidak ada release siap pakai; lihat onnx_parity.csv.')
    run.json({'classes':run.c['classes'],'labels_are':'integer indices in this order','score_is_calibrated':False},f'onnx/{selected["combo_id"]}/labels.json')
    run.json({'transform':info['preprocessing'],'implementation':'timm.data.create_transform(is_training=False)',
        'rgb':'EXIF transpose; alpha composite white; RGB','max_pixels':run.c['cleaning']['max_pixels'],
        'classifier_includes_scaler':name in ['logistic_regression','svm']},f'onnx/{selected["combo_id"]}/preprocessing.json')
    files=['extractor.onnx','classifier.onnx','labels.json','preprocessing.json']
    release={'cnn':cnn,'classifier':name,'parity_samples':len(table),'parity_passed':True,
        'exploratory':run.exploratory,'clinical_validation':False,
        'checksums':{n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in files}}
    run.json(release,f'onnx/{selected["combo_id"]}/release.json')
    print('ONNX diverifikasi:',folder);return folder
