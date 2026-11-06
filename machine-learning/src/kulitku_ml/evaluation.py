import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from .training import metrics
from .eda import gallery


def prediction_report(run,predictions,prefix):
    if predictions.empty:raise ValueError('Prediksi kosong.')
    y=predictions.y_true.to_numpy(dtype=int);p=predictions.y_pred.to_numpy(dtype=int)
    report=classification_report(y,p,labels=[0,1,2,3],target_names=run.c['classes'],output_dict=True,zero_division=0)
    run.table(pd.DataFrame(report).T.rename_axis('class').reset_index(),prefix+'_classification_report.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,norm in zip(axes,[None,'true']):
        ConfusionMatrixDisplay.from_predictions(y,p,labels=[0,1,2,3],display_labels=run.c['classes'],
            normalize=norm,xticks_rotation=30,ax=ax,colorbar=False,values_format='.2f' if norm else 'd')
        ax.set_title('Normalisasi per kelas' if norm else 'Jumlah')
    fig.suptitle(run.tag+prefix);run.figure(fig,prefix+'_confusion_matrices.png')
    perclass=pd.DataFrame([report[c] for c in run.c['classes']],index=run.c['classes'])
    fig,ax=plt.subplots(figsize=(10,4));perclass[['precision','recall','f1-score']].plot.bar(ax=ax,rot=20)
    ax.set(ylim=(0,1.05),title=run.tag+prefix);run.figure(fig,prefix+'_per_class.png')
    wrong=predictions[predictions.y_true.ne(predictions.y_pred)]
    gallery(run,wrong,prefix+'_errors.png',run.tag+'Contoh salah: '+prefix)
    return metrics(y,p)


def validation_report(run,cnn,name):
    prefix=f'validation_{cnn}_{name}'
    return prediction_report(run,run.read(f'oof_{cnn}_{name}.csv'),prefix)


def cv_overview(run):
    results=[]
    for cnn in run.c['extractors']:
        for name in run.c['classifiers']:
            path=run.out/'models'/cnn/name/'cv_summary.json'
            if not path.exists():raise ValueError(f'Belum selesai: {cnn}/{name}')
            results.append(json.loads(path.read_text()))
    t=pd.DataFrame(results);run.table(t,'cv_overview.csv')
    matrix=t.pivot(index='cnn',columns='classifier',values='mean_macro_f1')
    fig,ax=plt.subplots(figsize=(9,5));im=ax.imshow(matrix.values,vmin=0,vmax=1,cmap='YlGnBu')
    ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=20,ha='right')
    ax.set_yticks(range(len(matrix)),matrix.index)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):ax.text(j,i,f'{matrix.iloc[i,j]:.3f}',ha='center',va='center',color='white' if matrix.iloc[i,j]>.5 else 'black')
    fig.colorbar(im,ax=ax,label='Mean macro F1 CV');ax.set_title(run.tag+'Sembilan kombinasi')
    run.figure(fig,'cv_heatmap.png')
    fig,ax=plt.subplots(figsize=(13,5));ax.bar(t.combo_id,t.mean_macro_f1,yerr=t.std_macro_f1,capsize=4)
    ax.tick_params(axis='x',rotation=65,labelsize=8);ax.set(ylabel='Macro F1',ylim=(0,1.05),title=run.tag+'Mean ± SD antar fold')
    run.figure(fig,'cv_mean_std.png')
    fig,ax=plt.subplots(figsize=(12,4))
    for r in results:
        scores=run.read('cv_'+r['cnn']+'_'+r['classifier']+'.csv')
        # Find best candidate by actual aggregate, same deterministic rule as training.
        group=scores.groupby('candidate').macro_f1.mean().sort_index();winner=group.idxmax()
        fold_scores=scores[scores.candidate.eq(winner)].sort_values('fold')
        ax.plot(fold_scores.fold.astype(int)+1,fold_scores.macro_f1,marker='o',label=r['combo_id'])
    ax.set(xlabel='Fold',ylabel='Macro F1',title=run.tag+'Variasi antar fold');ax.legend(fontsize=7,bbox_to_anchor=(1.02,1))
    run.figure(fig,'cv_fold_scores.png');return t


def final_test(run,df,cnn,name):
    import joblib
    if not (run.out/'selection.json').exists():raise ValueError('Kunci pilihan sebelum membuka test.')
    selection=json.loads((run.out/'selection.json').read_text())
    if (cnn,name)!=(selection['cnn'],selection['classifier']):raise ValueError('Test utama hanya untuk kombinasi terkunci.')
    cached=run.out/'tabel/test_predictions.csv'
    if cached.exists():return prediction_report(run,run.read('test_predictions.csv'),'test_final')
    x=np.load(run.out/'features'/cnn/'features.npy',mmap_mode='r')
    model=joblib.load(run.out/'models'/cnn/name/'classifier.joblib')
    eligible=df.evaluation_eligible.astype(str).str.lower().eq('true')
    indices=np.flatnonzero((df.partition.eq('test')&eligible).to_numpy())
    y=df.label.map({c:i for i,c in enumerate(run.c['classes'])}).to_numpy()
    p=model.predict(x[indices])
    table=df.iloc[indices][['image_id','relative_path','label','split_group']].copy()
    table['y_true']=y[indices];table['y_pred']=p
    run.table(table,'test_predictions.csv')
    scores=prediction_report(run,table,'test_final')
    rng=np.random.default_rng(run.c['seed']);groups=table.split_group.unique();values=[];discarded=0
    group_rows={g:np.flatnonzero(table.split_group.to_numpy()==g) for g in groups}
    yt=table.y_true.to_numpy();yp=table.y_pred.to_numpy()
    for _ in range(run.c['evaluation']['bootstrap_repeats']):
        take=np.concatenate([group_rows[g] for g in rng.choice(groups,size=len(groups),replace=True)])
        if len(set(yt[take]))<4:discarded+=1;continue
        values.append(metrics(yt[take],yp[take])['macro_f1'])
    ci=np.quantile(values,[.025,.975]).tolist() if values else None
    run.json({**scores,'macro_f1_group_bootstrap_ci95':ci,'discarded_missing_class_replicates':discarded,
        'exploratory':run.exploratory,'patient_independence_verified':False},'test_metrics.json')
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(['CV development','Final test'],[selection['mean_macro_f1'],scores['macro_f1']])
    ax.set(ylim=(0,1.05),ylabel='Macro F1',title=run.tag+'CV vs test internal')
    run.figure(fig,'cv_vs_test.png');return scores
