import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold
from tqdm.auto import tqdm


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def split(run,cleaned):
    target=run.out/'tabel/splits.csv'
    if target.exists():return run.read('splits.csv')
    df=cleaned.copy()
    print('Hash dilakukan sekali pada tahap split, bukan pada EDA.')
    df['sha256']=[file_hash(run.root/p) for p in tqdm(df.relative_path,desc='Hash untuk exact duplicate')]
    conflict=df.groupby('sha256').label.transform('nunique').gt(1)
    if conflict.any():
        run.table(df[conflict],'duplicate_label_conflicts.csv')
        raise ValueError('Duplikat persis memiliki label berbeda. Review duplicate_label_conflicts.csv; isi review cleaning dan buat run baru.')
    if run.c['split']['mode']=='verified_groups':
        source=run.c['split']['lineage_csv']
        if not source:raise ValueError('Mode verified_groups memerlukan CSV lineage hasil review.')
        audit=pd.read_csv(source,dtype=str,keep_default_na=False)
        required={'relative_path','lineage_group','original_status','evidence'}
        if not required<=set(audit) or audit.relative_path.duplicated().any():raise ValueError('CSV lineage tidak valid.')
        df=df.drop(columns=['original_status','lineage_group']).merge(audit[list(required)],on='relative_path',how='left',validate='one_to_one')
        if df[['lineage_group','original_status','evidence']].isna().any().any() or df.lineage_group.eq('').any() or df.evidence.eq('').any():
            raise ValueError('Seluruh gambar yang dipakai memerlukan kelompok dan bukti terverifikasi.')
        if not df.original_status.isin(['verified_original','verified_augmented']).all():raise ValueError('Status original tidak valid.')
        if df.groupby('sha256').lineage_group.nunique().gt(1).any():raise ValueError('Exact duplicate melintasi lineage_group. Selesaikan audit kelompok.')
        df['split_group']=df.lineage_group
    else:
        print('PERINGATAN: eksperimen eksploratori. SHA hanya mencegah duplikat byte melintasi split; variasi augmentasi/pasien belum terpisah.')
        df['split_group']='sha_'+df.sha256
    # Prefer a verified original among byte-identical copies; deterministic path tie-break.
    df['_rank']=df.original_status.ne('verified_original').astype(int)
    df=df.sort_values(['_rank','relative_path'])
    removed=df[df.duplicated('sha256')].copy();run.table(removed,'exact_duplicates_removed_at_split.csv')
    df=df.drop_duplicates('sha256').drop(columns='_rank').sort_values('relative_path').reset_index(drop=True)
    y=df.label.map({c:i for i,c in enumerate(run.c['classes'])}).to_numpy()
    counts=df.groupby('label').split_group.nunique().reindex(run.c['classes'],fill_value=0)
    if counts.min()<run.c['split']['holdout_folds']:raise ValueError('Kelompok per kelas tidak cukup untuk holdout berstratifikasi.')
    # With one row per group, ordinary stratification preserves all groups exactly
    # and avoids unnecessary greedy group allocation on small fixtures.
    unique_groups=df.split_group.is_unique
    cv_class=StratifiedKFold if unique_groups else StratifiedGroupKFold
    cv=cv_class(n_splits=run.c['split']['holdout_folds'],shuffle=True,random_state=run.c['seed'])
    dev_idx,test_idx=next(cv.split(df,y) if unique_groups else cv.split(df,y,df.split_group))
    df['partition']='development';df.loc[test_idx,'partition']='test';df['fold']=-1
    dev=df.iloc[dev_idx]
    k=run.c['split']['folds']
    if dev.groupby('label').split_group.nunique().reindex(run.c['classes'],fill_value=0).min()<k:
        raise ValueError('Kelompok development per kelas tidak cukup untuk CV yang diminta.')
    inner=cv_class(n_splits=k,shuffle=True,random_state=run.c['seed'])
    iterator=inner.split(dev,y[dev_idx]) if unique_groups else inner.split(dev,y[dev_idx],dev.split_group)
    for fold,(_,val) in enumerate(iterator):
        df.loc[dev.index[val],'fold']=fold
    df['evaluation_eligible']=True if run.exploratory else df.original_status.eq('verified_original')
    labels=set(run.c['classes'])
    for fold in range(k):
        val=df[(df.partition=='development')&(df.fold==fold)&df.evaluation_eligible]
        train=df[(df.partition=='development')&(df.fold!=fold)]
        if set(val.label)!=labels or set(train.label)!=labels:raise ValueError('Ada fold kehilangan kelas; perbaiki kelompok/protokol, bukan memilih seed dari skor.')
        assert not set(val.split_group)&set(train.split_group)
    test=df[(df.partition=='test')&df.evaluation_eligible]
    if set(test.label)!=labels:raise ValueError('Test evaluable tidak memiliki empat kelas.')
    assert not set(df[df.partition=='development'].split_group)&set(df[df.partition=='test'].split_group)
    run.table(df,'splits.csv')
    run.table(df.groupby(['partition','label','fold','evaluation_eligible']).size().rename('n').reset_index(),'split_summary.csv')
    run.json({'mode':run.c['split']['mode'],'patient_independence_verified':False,
              'augmentation_lineage_verified':not run.exploratory,'removed_exact_duplicates':len(removed),
              'split_digest':hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()},'split_status.json')
    return df


def cv_indices(run,df):
    eligible=df.evaluation_eligible.astype(str).str.lower().eq('true').to_numpy()
    dev=df.partition.eq('development').to_numpy();fold=pd.to_numeric(df.fold).to_numpy()
    return [(np.flatnonzero(dev&(fold!=f)),np.flatnonzero(dev&(fold==f)&eligible)) for f in range(run.c['split']['folds'])]
