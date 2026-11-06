import json,time,warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.exceptions import ConvergenceWarning
from .splitting import cv_indices
from .config import digest


def classifier(run,name,params=None):
    seed=run.c['seed']
    if name=='logistic_regression':
        model=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=5000,random_state=seed))])
    elif name=='svm':model=Pipeline([('scale',StandardScaler()),('model',SVC(probability=False))])
    elif name=='random_forest':model=Pipeline([('model',RandomForestClassifier(random_state=seed,n_jobs=run.c['training']['jobs']))])
    else:raise ValueError(name)
    return model.set_params(**(params or {}))


def metrics(y,p):
    return {'macro_f1':f1_score(y,p,average='macro',labels=[0,1,2,3],zero_division=0),
        'accuracy':accuracy_score(y,p),'macro_precision':precision_score(y,p,average='macro',labels=[0,1,2,3],zero_division=0),
        'macro_recall':recall_score(y,p,average='macro',labels=[0,1,2,3],zero_division=0)}


def train_classifier(run,df,x,cnn,name):
    folder=run.out/'models'/cnn/name;folder.mkdir(parents=True,exist_ok=True)
    complete=folder/'cv_summary.json'
    identity=digest({'rows':df[['image_id','partition','fold']].to_dict('records'),'shape':list(x.shape),'grid':run.c['classifiers'][name]})
    if complete.exists():
        result=json.loads(complete.read_text())
        if result['identity']!=identity:raise ValueError('Cache training berbeda.')
        print('Selesai sebelumnya:',cnn,name);return result
    y=df.label.map({label:i for i,label in enumerate(run.c['classes'])}).to_numpy()
    folds=cv_indices(run,df)
    grid=list(ParameterGrid(run.c['classifiers'][name]))
    records=[]
    log=folder/'cv_progress.json'
    if log.exists():
        old=json.loads(log.read_text())
        if old['identity']!=identity:raise ValueError('Progress berasal dari grid/data berbeda.')
        records=old['records']
    done={(r['candidate'],r['fold']) for r in records}
    for candidate,params in enumerate(grid):
        for fold,(tr,va) in enumerate(folds):
            if (candidate,fold) in done:continue
            print(f'{cnn}/{name} kandidat {candidate+1}/{len(grid)} fold {fold+1}/{len(folds)}',flush=True)
            model=classifier(run,name,params);start=time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model.fit(x[tr],y[tr])
            elapsed=time.perf_counter()-start
            pred=model.predict(x[va]);record={'candidate':candidate,'fold':fold,'params':params,'fit_seconds':elapsed,**metrics(y[va],pred)}
            records.append(record)
            run.json({'identity':identity,'records':records},f'models/{cnn}/{name}/cv_progress.json')
    table=pd.DataFrame(records)
    run.table(table,f'cv_{cnn}_{name}.csv')
    ranking=table.groupby('candidate').agg(mean_macro_f1=('macro_f1','mean'),std_macro_f1=('macro_f1','std'),
        mean_accuracy=('accuracy','mean'),total_fit_seconds=('fit_seconds','sum')).reset_index().sort_values(['mean_macro_f1','candidate'],ascending=[False,True])
    winner=int(ranking.iloc[0].candidate);best_params=grid[winner]
    # Refit only development; test labels never influence tuning.
    dev=np.flatnonzero(df.partition.eq('development').to_numpy())
    best=classifier(run,name,best_params);start=time.perf_counter();best.fit(x[dev],y[dev]);refit=time.perf_counter()-start
    joblib.dump(best,folder/'classifier.joblib')
    # OOF validation uses selected hyperparameters; optimistic from selection, not final test.
    oof=[]
    for fold,(tr,va) in enumerate(folds):
        m=classifier(run,name,best_params);m.fit(x[tr],y[tr]);pred=m.predict(x[va])
        for i,p in zip(va,pred):oof.append({'image_id':df.iloc[i].image_id,'relative_path':df.iloc[i].relative_path,
            'label':df.iloc[i].label,'y_true':int(y[i]),'y_pred':int(p),'fold':fold})
    run.table(pd.DataFrame(oof),f'oof_{cnn}_{name}.csv')
    result={'identity':identity,'cnn':cnn,'classifier':name,'combo_id':cnn+'__'+name,
            'best_params':best_params,'mean_macro_f1':float(ranking.iloc[0].mean_macro_f1),
            'std_macro_f1':float(ranking.iloc[0].std_macro_f1),'refit_seconds':refit,
            'total_search_fit_seconds':float(table.fit_seconds.sum()),'exploratory':run.exploratory}
    run.json(result,f'models/{cnn}/{name}/cv_summary.json');return result
