import numpy as np
from PIL import Image, ImageOps


def rgb(path,max_pixels=25_000_000):
    with Image.open(path) as im:
        if im.width*im.height>max_pixels:raise ValueError('Batas piksel terlampaui.')
        im.seek(0);im.load();img=ImageOps.exif_transpose(im)
        if img.mode in ['RGBA','LA'] or 'transparency' in img.info:
            a=img.convert('RGBA');return Image.alpha_composite(Image.new('RGBA',a.size,'white'),a).convert('RGB')
        return img.convert('RGB')


def build_transform(spec):
    """Actual deterministic timm transform. Same implementation for extraction/export parity."""
    from timm.data import create_transform
    return create_transform(input_size=tuple(spec['input_size']),interpolation=spec['interpolation'],
        mean=tuple(spec['mean']),std=tuple(spec['std']),crop_pct=spec['crop_pct'],
        crop_mode=spec.get('crop_mode','center'),is_training=False)


def transform_spec(model):
    from timm.data import resolve_model_data_config
    c=resolve_model_data_config(model)
    return {k:c[k] for k in ['input_size','interpolation','mean','std','crop_pct','crop_mode'] if k in c}
