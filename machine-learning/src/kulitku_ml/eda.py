import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from .preprocessing import rgb


def gallery(run,df,name,title,n=8):
    rows=df.head(n)
    if rows.empty:return
    fig,axes=plt.subplots((len(rows)+3)//4,4,figsize=(13,3.4*((len(rows)+3)//4)),squeeze=False)
    for ax in axes.ravel():ax.axis('off')
    for ax,r in zip(axes.ravel(),rows.itertuples()):
        try:
            im=rgb(run.root/r.relative_path);im.thumbnail((400,400));ax.imshow(im)
            ax.set_title(f'{r.label} | {r.image_id}',fontsize=8)
        except Exception as e:ax.text(.1,.5,type(e).__name__)
    fig.suptitle(title);run.figure(fig,name)


def eda(run,df,stage):
    counts=df.label.value_counts().reindex(run.c['classes'],fill_value=0)
    table=counts.rename('n').rename_axis('label').reset_index();table['percent']=table.n/max(len(df),1)*100
    run.table(table,f'{stage}_class_counts.csv')
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(table.label,table.n,color=['#2563eb','#14b8a6','#8b5cf6','#f59e0b'])
    ax.bar_label(ax.containers[0]);ax.set(title=f'{stage}: jumlah gambar/file',ylabel='Jumlah');ax.tick_params(axis='x',rotation=15)
    run.figure(fig,f'{stage}_counts.png')
    run.table(pd.crosstab(df.label,df.status).reset_index(),f'{stage}_readability.csv')
    run.table(df.groupby(['label','format','mode'],dropna=False).size().rename('count').reset_index(),f'{stage}_formats_modes.csv')
    numeric=df.copy()
    for col in ['width','height','bytes']:numeric[col]=pd.to_numeric(numeric[col],errors='coerce')
    numeric['megapixels']=numeric.width*numeric.height/1e6;numeric['aspect_ratio']=numeric.width/numeric.height
    run.table(numeric.groupby('label')[['width','height','megapixels','aspect_ratio','bytes']].agg(['count','min','median','mean','max']).reset_index(),f'{stage}_dimensions_summary.csv')
    valid=numeric[numeric.status.eq('readable')].copy()
    if len(valid):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for label in run.c['classes']:
            s=valid[valid.label.eq(label)]
            axes[0,0].hist(s.megapixels,bins=30,alpha=.4,label=label)
            axes[0,1].hist(s.aspect_ratio,bins=30,alpha=.4,label=label)
            s=s.sample(min(1000,len(s)),random_state=run.c['seed'])
            axes[1,0].scatter(s.width,s.height,s=5,alpha=.3,label=label)
        axes[0,0].set(xlabel='Megapiksel',ylabel='Jumlah');axes[0,0].legend(fontsize=7)
        axes[0,1].set(xlabel='Rasio lebar/tinggi',ylabel='Jumlah')
        axes[1,0].set(xlabel='Lebar',ylabel='Tinggi',xscale='log',yscale='log')
        axes[1,1].hist(valid.bytes/1024,bins=40);axes[1,1].set(xlabel='KiB',ylabel='Jumlah')
        run.figure(fig,f'{stage}_dimensions.png')
        mode=pd.crosstab(valid.label,valid['mode']);fig,ax=plt.subplots(figsize=(9,4));mode.plot.bar(stacked=True,ax=ax,rot=15)
        ax.set_title('Mode warna sumber');run.figure(fig,f'{stage}_color_modes.png')
    rows=[]
    for i,label in enumerate(run.c['classes']):
        pool=valid[valid.label.eq(label)]
        sample=pool.sample(min(len(pool),run.c['eda']['sample_per_class']),random_state=run.c['seed'])
        gallery(run,sample,f'{stage}_gallery_{i}.png',f'{stage}: {label}',run.c['eda']['gallery_per_class'])
        for r in tqdm(sample.itertuples(),total=len(sample),desc=f'EDA sampel {label}'):
            im=rgb(run.root/r.relative_path);im.thumbnail((256,256))
            gray=np.asarray(im.convert('L'),dtype=np.float32);arr=np.asarray(im)
            p=np.bincount(gray.astype('uint8').ravel(),minlength=256).astype(float);p=p[p>0]/p.sum()
            lap=(-4*gray[1:-1,1:-1]+gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:])
            record={'image_id':r.image_id,'relative_path':r.relative_path,'label':r.label,
                'brightness':float(gray.mean()),'contrast':float(gray.std()),
                'saturation':float(np.asarray(im.convert('HSV'))[:,:,1].mean()/255),
                'entropy':float(-(p*np.log2(p)).sum()),'laplacian_variance':float(lap.var()) if lap.size else 0.}
            for ch in range(3):
                h=np.histogram(arr[:,:,ch],bins=32,range=(0,256))[0].astype(float);h/=h.sum()
                for j,v in enumerate(h):record[f'h{ch}_{j}']=v
            rows.append(record)
    quality=pd.DataFrame(rows)
    if len(quality):
        run.table(quality,f'{stage}_sample_quality.csv')
        columns=['brightness','contrast','saturation','entropy','laplacian_variance']
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for ax,col in zip(axes.ravel(),columns):
            ax.boxplot([quality.loc[quality.label.eq(c),col].values for c in run.c['classes']],labels=run.c['classes'],showfliers=False)
            ax.set_title(col);ax.tick_params(axis='x',rotation=30,labelsize=8)
        axes.ravel()[-1].axis('off');fig.suptitle('Kualitas visual SAMPEL; bukan ambang eksklusi')
        run.figure(fig,f'{stage}_quality_sample.png')
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for ax,label in zip(axes.ravel(),run.c['classes']):
            s=quality[quality.label.eq(label)]
            for ch,color in enumerate(['red','green','blue']):
                if len(s):ax.plot(np.arange(32)*8+4,s[[f'h{ch}_{j}' for j in range(32)]].mean().values,color=color)
            ax.set(title=f'{label}; n={len(s)}',xlabel='Intensitas',ylabel='Proporsi piksel rata-rata')
        run.figure(fig,f'{stage}_rgb_histograms_sample.png')
        gallery(run,quality.nsmallest(8,'laplacian_variance'),f'{stage}_low_laplacian.png','Laplacian rendah: review, bukan otomatis dibuang')
    return table


def compare(run,before,after):
    rows=[]
    for label in run.c['classes']:
        rows.append({'label':label,'before':int(before.label.eq(label).sum()),'after':int(after.label.eq(label).sum())})
    t=pd.DataFrame(rows);run.table(t,'cleaning_comparison.csv')
    fig,ax=plt.subplots(figsize=(10,4));t.set_index('label').plot.bar(ax=ax,rot=15)
    ax.set_title('Sebelum vs sesudah cleaning teknis/manual');run.figure(fig,'cleaning_comparison.png')
    return t
