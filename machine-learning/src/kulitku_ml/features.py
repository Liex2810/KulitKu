import gc, json, random, hashlib
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from .config import digest
from .preprocessing import rgb, build_transform, transform_spec


def load_extractor(run,name):
    import timm, torch
    spec=run.c['extractors'][name]
    model=timm.create_model(spec['id'],pretrained=spec['pretrained'],num_classes=0,global_pool='avg')
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def restore_extractor(run,name):
    import timm,torch
    folder=run.out/'features'/name
    info=json.loads((folder/'metadata.json').read_text())
    model=timm.create_model(info['model_id'],pretrained=False,num_classes=0,global_pool='avg')
    model.load_state_dict(torch.load(folder/'extractor.pt',map_location='cpu',weights_only=True))
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,info


def extract_features(run,df,name):
    import torch
    from torch.utils.data import Dataset,DataLoader
    folder=run.out/'features'/name;folder.mkdir(parents=True,exist_ok=True)
    rows_digest=digest(df[['image_id','relative_path','label','partition','fold']].to_dict('records'))
    meta=folder/'metadata.json';target=folder/'features.npy'
    if meta.exists() and target.exists():
        info=json.loads(meta.read_text())
        if info['rows_digest']!=rows_digest:raise ValueError('Urutan/data fitur berubah. Gunakan run baru.')
        x=np.load(target,mmap_mode='r')
        if x.shape!=tuple(info['shape']):raise ValueError('Cache fitur tidak lengkap.')
        print('Pakai fitur tersimpan:',name);return x
    seed=run.c['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    model=load_extractor(run,name)
    spec=transform_spec(model);transform=build_transform(spec)
    class Images(Dataset):
        def __len__(self):return len(df)
        def __getitem__(self,i):return transform(rgb(run.root/df.iloc[i].relative_path,run.c['cleaning']['max_pixels']))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=model.to(device)
    loader=DataLoader(Images(),batch_size=run.c['features']['batch_size'],shuffle=False,num_workers=0)
    pieces=[]
    with torch.inference_mode():
        for batch in tqdm(loader,desc='Fitur '+name):pieces.append(model(batch.to(device)).cpu().numpy().astype('float32'))
    x=np.concatenate(pieces,axis=0)
    if x.ndim!=2 or len(x)!=len(df) or not np.isfinite(x).all():raise ValueError('Fitur tidak valid.')
    temp=target.with_suffix('.partial')
    with temp.open('wb') as f:np.save(f,x)
    temp.replace(target)
    model=model.cpu();torch.save(model.state_dict(),folder/'extractor.pt')
    info={'model_id':run.c['extractors'][name]['id'],'pretrained':run.c['extractors'][name]['pretrained'],
          'frozen':True,'preprocessing':spec,'shape':list(x.shape),'rows_digest':rows_digest,
          'weights_sha256':hashlib.sha256((folder/'extractor.pt').read_bytes()).hexdigest()}
    run.json(info,f'features/{name}/metadata.json')
    run.table(df[['image_id','relative_path','label','partition','fold']],f'features_order_{name}.csv')
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return np.load(target,mmap_mode='r')
