import pandas as pd


def clean(run,before):
    df=before.copy();df['source_label']=df.label
    df['keep']=df.status.eq('readable')&pd.to_numeric(df.n_frames,errors='coerce').eq(1)
    df['reason']=''
    df.loc[~df.keep,'reason']='technical: '+df.loc[~df.keep,'status']
    df.loc[df.status.eq('readable')&~df.keep,'reason']='quarantine_multiframe'
    if run.c['cleaning']['exclude_below_minimum']:
        low=df.below_minimum.astype(str).str.lower().eq('true')
        df.loc[low,'keep']=False;df.loc[low,'reason']='minimum: '+run.c['cleaning']['minimum_reason']
    template=df[['image_id','relative_path','label','status']].copy()
    for k in ['decision','reason','reviewer','label_override','label_evidence']:template[k]=''
    run.table(template,'manual_review_template.csv')
    source=run.c['cleaning']['manual_review_csv']
    if source:
        review=pd.read_csv(source,dtype=str,keep_default_na=False)
        required={'image_id','relative_path','decision','reason','reviewer','label_override','label_evidence'}
        if not required<=set(review):raise ValueError('Kolom review tidak lengkap.')
        review=review[review.decision.ne('')|review.label_override.ne('')]
        if review.image_id.duplicated().any():raise ValueError('Review duplikat.')
        for r in review.itertuples():
            idx=df.index[df.image_id.eq(r.image_id)]
            if len(idx)!=1 or df.at[idx[0],'relative_path']!=r.relative_path:raise ValueError('Review tidak cocok dengan file.')
            i=idx[0]
            if r.decision not in ['keep','exclude'] or not r.reason.strip() or not r.reviewer.strip():raise ValueError('Keputusan/alasan/reviewer wajib.')
            if r.decision=='keep' and not df.at[i,'keep']:raise ValueError('Manual keep tidak melewati masalah teknis.')
            if r.label_override:
                if r.label_override not in run.c['classes'] or not r.label_evidence.strip():raise ValueError('Label baru membutuhkan bukti.')
                df.at[i,'label']=r.label_override
            if r.decision=='exclude':df.at[i,'keep']=False
            df.at[i,'reason']='manual '+r.decision+': '+r.reason
            df.at[i,'reviewer']=r.reviewer;df.at[i,'label_evidence']=r.label_evidence
    df['original_status']='unknown';df['lineage_group']=''
    run.table(df,'cleaning_decisions.csv');run.table(df[~df.keep],'excluded.csv')
    after=df[df.keep].reset_index(drop=True)
    if after.empty:raise ValueError('Tidak ada gambar lolos; keputusan tersimpan di CSV.')
    run.table(after,'cleaned.csv');return after
