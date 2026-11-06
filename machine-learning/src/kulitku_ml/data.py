from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os, warnings
import pandas as pd
from PIL import Image, ImageFile
from tqdm.auto import tqdm


def inventory(run,force=False):
    path=run.out/'tabel/inventory.csv'
    if path.exists() and not force:return run.read('inventory.csv')
    rows=[]
    for label in run.c['classes']:
        if not (run.root/label).is_dir():raise FileNotFoundError(f'Folder kelas tidak ada: {run.root/label}')
        print('Daftar file:',label,flush=True)
        for folder,dirs,files in os.walk(run.root/label,followlinks=False):
            dirs[:]=sorted(d for d in dirs if not (Path(folder)/d).is_symlink())
            for name in sorted(files):
                p=Path(folder)/name
                rows.append({'relative_path':p.relative_to(run.root).as_posix(),'label':label})
    if not rows:raise ValueError('Dataset kosong.')
    rows=sorted(rows,key=lambda r:r['relative_path'])
    for i,r in enumerate(rows):r['image_id']=f'img_{i:07d}'
    c=run.c['cleaning'];ImageFile.LOAD_TRUNCATED_IMAGES=False
    Image.MAX_IMAGE_PIXELS=c['max_pixels']
    warnings.filterwarnings('error',category=Image.DecompressionBombWarning)
    def inspect(row):
        r={**row,'width':None,'height':None,'format':'','mode':'','n_frames':None,
           'bytes':None,'mtime_ns':None,'status':'pending','below_minimum':False,
           'alpha':False,'extension_mismatch':False,'error':''}
        p=run.root/r['relative_path']
        if p.is_symlink():r['status']='symlink';return r
        try:
            st=p.stat();r.update(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
            ext=p.suffix.lower()
            if ext not in ['.jpg','.jpeg','.png','.webp']:r['status']='unsupported_extension';return r
            with Image.open(p) as im:
                w,h=im.size
                r.update(width=w,height=h,format=im.format,mode=im.mode,n_frames=getattr(im,'n_frames',1),
                         below_minimum=min(w,h)<c['min_side'],alpha=im.mode in ['RGBA','LA'] or 'transparency' in im.info,
                         extension_mismatch=im.format!={'.jpg':'JPEG','.jpeg':'JPEG','.png':'PNG','.webp':'WEBP'}[ext])
                if w*h>c['max_pixels']:r['status']='pixel_limit';return r
                if im.format not in ['JPEG','PNG','WEBP']:r['status']='unsupported_content';return r
                im.seek(0);im.load();im.convert('RGB');r['status']='readable'
        except (Image.DecompressionBombWarning,Image.DecompressionBombError):r['status']='pixel_limit'
        except Exception as e:r['status']='decode_error';r['error']=type(e).__name__+': '+str(e)[:120]
        return r
    with ThreadPoolExecutor(max_workers=c['workers']) as pool:
        df=pd.DataFrame(tqdm(pool.map(inspect,rows),total=len(rows),desc='Baca gambar; tanpa hash'))
    run.table(df,'inventory.csv');return df
