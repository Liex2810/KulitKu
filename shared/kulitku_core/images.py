from io import BytesIO
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

LABELS = ["Eczema", "Herpes Zoster", "Normal", "Ringworm"]

def save_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")

def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def load_rgb(source):
    try:
        with Image.open(source) as image:
            if image.width*image.height>25_000_000:raise ValueError("Maximum image size is 25 megapixels")
            if image.format not in {"JPEG","PNG","WEBP"}:raise ValueError("Use JPEG, PNG or WEBP")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError,Image.DecompressionBombError,OSError) as e:
        raise ValueError("Invalid image") from e

def preprocess(image,spec):
    methods={"bilinear":Image.Resampling.BILINEAR,"bicubic":Image.Resampling.BICUBIC,"lanczos":Image.Resampling.LANCZOS}
    image=image.resize((spec["size"],spec["size"]),methods[spec["interpolation"]])
    x=np.asarray(image,dtype=np.float32)/255.0
    x=(x-np.asarray(spec["mean"],np.float32))/np.asarray(spec["std"],np.float32)
    return np.ascontiguousarray(x.transpose(2,0,1)[None],dtype=np.float32)
