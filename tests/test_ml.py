import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from kulitku_core.images import LABELS
from kulitku_ml.data import checked,split,validate_frozen
from kulitku_ml.benchmark import folds,pipeline

def dataset(groups=9):
    rows=[]
    for k,label in enumerate(LABELS):
        for g in range(groups):
            for original in [1,0]:
                rows.append({"path":f"{label}/{g}-{original}.png","label":label,"target":k,"sha256":f"hash-{k}-{g}-{original}",
                    "phash":"00","width":128,"height":128,"source":"synthetic","group_id":f"patient-{k}-{g}","is_original":original,"include":1})
    return pd.DataFrame(rows)

def test_audit_is_required():
    frame=dataset();frame.loc[0,"group_id"]=""
    with pytest.raises(ValueError,match="group_id"):checked(frame)

def test_unknown_original_is_not_silently_accepted():
    frame=dataset();frame["is_original"]="unknown"
    with pytest.raises(ValueError,match="is_original"):checked(frame)

def test_duplicates_cannot_cross_groups():
    frame=dataset();frame.loc[2,"sha256"]=frame.loc[0,"sha256"]
    with pytest.raises(ValueError,match="duplicates"):checked(frame)

def test_group_split_and_fold_original_validation(tmp_path):
    f=dataset();source=tmp_path/"review.csv";f.to_csv(source,index=False)
    cfg=tmp_path/"config.json";cfg.write_text(json.dumps({"folds":3,"test_folds":3,"seed":42}))
    split(source,tmp_path/"split",cfg,True)
    frozen=validate_frozen(pd.read_csv(tmp_path/"split/manifest.csv"))
    assert frozen.groupby("group_id").split.nunique().max()==1
    assert frozen[frozen.split=="test"].is_original.eq(1).all()
    development=frozen[frozen.split=="development"].reset_index(drop=True)
    for train,val in folds(development):
        assert set(development.iloc[train].group_id).isdisjoint(development.iloc[val].group_id)
        assert development.iloc[val].is_original.eq(1).all()

def test_checked_preserves_order_after_freeze():
    first=checked(dataset());second=checked(first)
    assert first.path.tolist()==second.path.tolist()

def test_pipeline_has_scaler_only_when_needed():
    assert "scaler" in pipeline("lr",42).named_steps
    assert "scaler" in pipeline("svm",42).named_steps
    assert "scaler" not in pipeline("rf",42).named_steps
