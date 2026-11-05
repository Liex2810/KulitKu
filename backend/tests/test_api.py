from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.settings import Settings
from app.pdf import build_pdf
from pypdf import PdfReader

def image_bytes():
    stream=BytesIO();Image.fromarray(np.random.default_rng(0).integers(0,255,(128,128,3),dtype=np.uint8)).save(stream,format="PNG")
    return stream.getvalue()

@pytest.fixture
def client():
    with TestClient(create_app(Settings(auth_mode="local",persist_history=False,demo_mode=True))) as c:yield c

def test_demo_never_returns_medical_prediction(client):
    r=client.post("/v1/scans",files={"file":("a.png",image_bytes())})
    assert r.status_code==200 and r.json()["status"]=="demo_only"
    assert r.json()["predicted_class"] is None and not r.json()["saved"]

def test_invalid_upload(client):
    assert client.post("/v1/scans",files={"file":("a.png",b"bad")}).status_code==400

def test_oversized_upload():
    with TestClient(create_app(Settings(auth_mode="local",persist_history=False,demo_mode=True,max_upload_bytes=10))) as c:
        assert c.post("/v1/scans",files={"file":("a.png",b"x"*11)}).status_code==413

def test_local_history_disabled(client):
    assert client.get("/v1/scans").json()==[]
    assert client.get("/v1/diaries").json()==[]
    assert client.post("/v1/diaries",json={"title":"Example"}).status_code==503

def test_auth_is_required():
    with TestClient(create_app(Settings(supabase_url="https://example.supabase.co",supabase_anon_key="test",demo_mode=True))) as c:
        assert c.get("/v1/scans").status_code==401
        assert c.get("/v1/diaries").status_code==401

def test_pdf_escapes_notes_and_embeds_photo():
    pdf=build_pdf({"title":"Diary <test>","notes":"First & second"},[{"id":"1","created_at":"2026-11-05","status":"classified","predicted_class":"Eczema","model_version":"test"}],{"1":image_bytes()})
    reader=PdfReader(BytesIO(pdf));text="".join(p.extract_text() for p in reader.pages)
    assert "First & second" in text and "Eczema" in text and "not a medical diagnosis" in text

def test_diary_access_uses_verified_owner(monkeypatch):
    with TestClient(create_app(Settings(supabase_url="https://example.supabase.co",supabase_anon_key="test",demo_mode=True))) as c:
        store=c.app.state.store;monkeypatch.setattr(store,"user",lambda token:"verified-user")
        calls=[];monkeypatch.setattr(store,"rows",lambda token,table,user,**kw:calls.append((table,user)) or [])
        assert c.get("/v1/diaries",headers={"Authorization":"Bearer token"}).status_code==200
        assert calls==[("diaries","verified-user")]


def test_link_checks_both_owners(monkeypatch):
    from app.store import Store
    from fastapi import HTTPException
    store=Store(Settings(supabase_url='https://example.supabase.co',supabase_anon_key='test',demo_mode=True))
    seen=[]
    def lookup(token,table,user,id):
        seen.append((table,user,id))
        if table=='scans':raise HTTPException(404,'Not yours')
        return {'id':id}
    monkeypatch.setattr(store,'get',lookup)
    with pytest.raises(HTTPException) as exc:store.link('token','owner-a','diary-a','scan-b')
    assert exc.value.status_code==404
    assert seen==[('diaries','owner-a','diary-a'),('scans','owner-a','scan-b')]


def test_unlink_does_not_delete_scan(monkeypatch):
    from app.store import Store
    store=Store(Settings(supabase_url='https://example.supabase.co',supabase_anon_key='test',demo_mode=True))
    monkeypatch.setattr(store,'get',lambda *args:{'id':'diary'})
    calls=[]
    monkeypatch.setattr(store,'request',lambda method,path,token,**kwargs:calls.append((method,path,kwargs)))
    store.unlink('token','owner','diary','scan')
    assert len(calls)==1 and calls[0][1]=='/rest/v1/diary_scans'
    assert calls[0][2]['params']['user_id']=='eq.owner'


def test_pdf_endpoint_uses_linked_scans_and_filters_dates(monkeypatch):
    from uuid import uuid4
    id=str(uuid4())
    with TestClient(create_app(Settings(supabase_url='https://example.supabase.co',supabase_anon_key='test',demo_mode=True))) as c:
        store=c.app.state.store
        monkeypatch.setattr(store,'user',lambda token:'owner')
        monkeypatch.setattr(store,'get',lambda token,table,user,id:{'id':id,'title':'Example','notes':'Notes'})
        monkeypatch.setattr(store,'linked',lambda token,user,id:[{'id':'old','created_at':'2026-10-01T00:00:00Z','status':'classified','predicted_class':'Ringworm','model_version':'test'},
                                                             {'id':'new','created_at':'2026-11-01T00:00:00Z','status':'classified','predicted_class':'Eczema','model_version':'test'}])
        auth={'Authorization':'Bearer token'}
        r=c.get(f'/v1/diaries/{id}/pdf?date_from=2026-11-01',headers=auth)
        assert r.status_code==200 and r.headers['content-type']=='application/pdf'
        text=''.join(p.extract_text() for p in PdfReader(BytesIO(r.content)).pages)
        assert 'Eczema' in text and 'Ringworm' not in text
        assert c.get(f'/v1/diaries/{id}/pdf?date_from=2026-11-05&date_to=2026-11-01',headers=auth).status_code==400
