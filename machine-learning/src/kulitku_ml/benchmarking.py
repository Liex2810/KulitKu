import gc,json,time,platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from .features import restore_extractor
from .preprocessing import rgb,build_transform


def benchmark(run,df):
    import torch
    if (run.out/'selection.json').exists():
        print('Pilihan sudah terkunci; benchmark tidak mengubahnya. Gunakan run baru untuk protokol berbeda.')
        return run.read('benchmark.csv')
    records=[]
    eligible=df.evaluation_eligible.astype(str).str.lower().eq('true')
    pool=df[df.partition.eq('development')&eligible]
    samples=pd.concat([pool[pool.label.eq(c)].sample(min(run.c['benchmark']['samples_per_class'],int(pool.label.eq(c).sum())),
        random_state=run.c['seed']) for c in run.c['classes']])
    if samples.empty:raise ValueError('Sampel benchmark kosong.')
    run.table(samples[['image_id','relative_path','label']],'benchmark_sample.csv')
    torch.set_num_threads(run.c['benchmark']['threads'])
    raw=[]
    for cnn in run.c['extractors']:
        model,info=restore_extractor(run,cnn);transform=build_transform(info['preprocessing'])
        for name in run.c['classifiers']:
            summary=json.loads((run.out/'models'/cnn/name/'cv_summary.json').read_text())
            pipeline=joblib.load(run.out/'models'/cnn/name/'classifier.joblib')
            if name=='random_forest':pipeline.named_steps['model'].n_jobs=1
            def infer(path):
                batch=transform(rgb(path,run.c['cleaning']['max_pixels'])).unsqueeze(0)
                with torch.inference_mode():feature=model(batch).numpy().astype('float32')
                return pipeline.predict(feature)
            for _ in range(run.c['benchmark']['warmup']):infer(run.root/samples.iloc[0].relative_path)
            durations=[]
            for _ in range(run.c['benchmark']['repeats']):
                for r in samples.itertuples():
                    start=time.perf_counter();infer(run.root/r.relative_path);ms=(time.perf_counter()-start)*1000
                    durations.append(ms);raw.append({'combo_id':summary['combo_id'],'image_id':r.image_id,'milliseconds':ms})
            size=(run.out/'features'/cnn/'extractor.pt').stat().st_size+(run.out/'models'/cnn/name/'classifier.joblib').stat().st_size
            records.append({**summary,'median_ms':float(np.median(durations)),'p95_ms':float(np.quantile(durations,.95)),
                            'combined_bytes':size,'runtime':'PyTorch CPU + sklearn; batch1; RF predict jobs1'})
            run.table(pd.DataFrame(records),'benchmark_partial.csv')
        del model;gc.collect()
    table=pd.DataFrame(records);run.table(table,'benchmark.csv');run.table(pd.DataFrame(raw),'benchmark_timings.csv')
    top=table.mean_macro_f1.max();table['quality_candidate']=table.mean_macro_f1>=top-run.c['selection']['macro_f1_tolerance']
    winner=table[table.quality_candidate].sort_values(['median_ms','combined_bytes','combo_id']).iloc[0].to_dict()
    winner['selection_rule']='CV macroF1 within tolerance, then native CPU median latency, then bytes, then combo_id'
    run.table(table,'benchmark.csv');run.json(winner,'selection.json')
    run.json({'cpu':platform.processor(),'platform':platform.platform(),'torch_threads':torch.get_num_threads(),
        'scope':'RGB decode + timm transform + frozen CNN + scaler/classifier; model load/network excluded',
        'runtime':'PyTorch CPU + sklearn, RF prediction jobs1','network_included':False},'benchmark_environment.json')
    fig,ax=plt.subplots(figsize=(10,6))
    for r in table.itertuples():
        ax.scatter(r.median_ms,r.mean_macro_f1,s=140 if r.combo_id==winner['combo_id'] else 70,
            marker='*' if r.combo_id==winner['combo_id'] else 'o',label=r.combo_id)
    ax.legend(fontsize=7,bbox_to_anchor=(1.02,1),loc='upper left')
    ax.axhline(top-run.c['selection']['macro_f1_tolerance'],ls='--',color='gray')
    ax.set(xlabel='Median inferensi CPU native (ms)',ylabel='Mean macro F1 CV',title=run.tag+'Kualitas vs latensi')
    run.figure(fig,'benchmark_quality_latency.png')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    axes[0].bar(table.combo_id,table.total_search_fit_seconds);axes[0].set(ylabel='Detik',title='Total fit pencarian CV')
    axes[1].bar(table.combo_id,table.combined_bytes/2**20);axes[1].set(ylabel='MiB',title='Ukuran CNN + classifier native')
    for ax in axes:ax.tick_params(axis='x',rotation=75,labelsize=7)
    run.figure(fig,'benchmark_training_size.png');return table
