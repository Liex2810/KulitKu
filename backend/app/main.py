from contextlib import asynccontextmanager
from io import BytesIO
from time import perf_counter
from uuid import UUID,uuid4
from datetime import date
from fastapi import FastAPI,Depends,HTTPException,UploadFile,File,Form,Response
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials
from pydantic import BaseModel,Field
from starlette.concurrency import run_in_threadpool
from app.settings import Settings
from app.store import Store
from app.engine import Engine
from app.pdf import build_pdf
from kulitku_core.images import load_rgb

class DiaryInput(BaseModel):
    title:str=Field(min_length=1,max_length=150)
    notes:str=Field(default="",max_length=20000)

class LinkInput(BaseModel):
    scan_id:UUID

def create_app(settings=None,engine=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.settings=settings or Settings();app.state.engine=engine or Engine(app.state.settings)
        app.state.store=Store(app.state.settings);yield
    app=FastAPI(title="KulitKu API",version="0.2.0",lifespan=lifespan)
    bearer=HTTPBearer(auto_error=False)
    async def identity(c:HTTPAuthorizationCredentials|None=Depends(bearer)):
        if app.state.settings.auth_mode=="local":return "local-user",""
        if not c or c.scheme.lower()!="bearer":raise HTTPException(401,"Login required")
        return await run_in_threadpool(app.state.store.user,c.credentials),c.credentials
    def require_store():
        if not app.state.settings.persist_history:raise HTTPException(503,"Supabase persistence disabled in local demo")
    @app.get("/health")
    def health():return {"status":"ok",**app.state.engine.release,"auth_mode":app.state.settings.auth_mode}
    @app.post("/v1/scans")
    async def scan(file:UploadFile=File(...),save_image:bool=Form(True),who=Depends(identity)):
        begin=perf_counter();payload=bytearray()
        try:
            while chunk:=await file.read(64*1024):
                payload.extend(chunk)
                if len(payload)>app.state.settings.max_upload_bytes:raise HTTPException(413,"Maximum upload size exceeded")
        finally:await file.close()
        try:image=await run_in_threadpool(load_rgb,BytesIO(payload))
        except ValueError as e:raise HTTPException(400,str(e)) from e
        result=await run_in_threadpool(app.state.engine.predict,image)
        result.update(scan_id=str(uuid4()),inference_ms=round((perf_counter()-begin)*1000,2),saved=False,image_saved=False)
        if app.state.settings.persist_history and not result["synthetic_demo"]:
            user,token=who;path=None
            if save_image:
                content=BytesIO();image.save(content,format="JPEG",quality=90);path=f"{user}/{result['scan_id']}.jpg"
                await run_in_threadpool(app.state.store.upload,token,path,content.getvalue())
            row={"id":result["scan_id"],"user_id":user,"image_path":path,"status":result["status"],"predicted_class":result["predicted_class"],
                "probabilities":result["probabilities"],"quality_flags":result["quality_flags"],"model_version":result["model_version"],"inference_ms":result["inference_ms"]}
            try:await run_in_threadpool(app.state.store.insert,token,"scans",row)
            except HTTPException:
                if path:await run_in_threadpool(app.state.store.remove_photo,token,path)
                raise
            result.update(saved=True,image_saved=bool(path))
        return result
    @app.get("/v1/scans")
    def scans(who=Depends(identity)):
        if not app.state.settings.persist_history:return []
        return app.state.store.rows(who[1],"scans",who[0],order="created_at.desc",limit="100")
    @app.get("/v1/scans/{id}")
    def scan_detail(id:UUID,who=Depends(identity)):
        require_store();return app.state.store.get(who[1],"scans",who[0],str(id))
    @app.delete("/v1/scans/{id}")
    def delete_scan(id:UUID,who=Depends(identity)):
        require_store();app.state.store.delete(who[1],"scans",who[0],str(id));return {"deleted":True}
    @app.get("/v1/diaries")
    def diaries(who=Depends(identity)):
        if not app.state.settings.persist_history:return []
        return app.state.store.rows(who[1],"diaries",who[0],order="created_at.desc",limit="100")
    @app.post("/v1/diaries")
    def add_diary(body:DiaryInput,who=Depends(identity)):
        require_store();row={"id":str(uuid4()),"user_id":who[0],**body.model_dump()}
        app.state.store.insert(who[1],"diaries",row);return row
    @app.get("/v1/diaries/{id}")
    def diary_detail(id:UUID,who=Depends(identity)):
        require_store();row=app.state.store.get(who[1],"diaries",who[0],str(id))
        return {**row,"scans":app.state.store.linked(who[1],who[0],str(id))}
    @app.patch("/v1/diaries/{id}")
    def edit_diary(id:UUID,body:DiaryInput,who=Depends(identity)):
        require_store();app.state.store.update(who[1],"diaries",who[0],str(id),body.model_dump());return {"updated":True}
    @app.delete("/v1/diaries/{id}")
    def delete_diary(id:UUID,who=Depends(identity)):
        require_store();app.state.store.delete(who[1],"diaries",who[0],str(id));return {"deleted":True}
    @app.post("/v1/diaries/{id}/scans")
    def link(id:UUID,body:LinkInput,who=Depends(identity)):
        require_store();app.state.store.link(who[1],who[0],str(id),str(body.scan_id));return {"linked":True}
    @app.delete("/v1/diaries/{id}/scans/{scan_id}")
    def unlink(id:UUID,scan_id:UUID,who=Depends(identity)):
        require_store();app.state.store.unlink(who[1],who[0],str(id),str(scan_id));return {"unlinked":True}
    @app.get("/v1/diaries/{id}/pdf")
    def pdf(id:UUID,date_from:date|None=None,date_to:date|None=None,who=Depends(identity)):
        require_store()
        if date_from and date_to and date_from>date_to:raise HTTPException(400,"Invalid date range")
        diary=app.state.store.get(who[1],"diaries",who[0],str(id));scans=app.state.store.linked(who[1],who[0],str(id))
        scans=[s for s in scans if (not date_from or date.fromisoformat(s["created_at"][:10])>=date_from) and (not date_to or date.fromisoformat(s["created_at"][:10])<=date_to)]
        photos={}
        for s in scans:
            if s.get("image_path"):
                try:photos[s["id"]]=app.state.store.photo(who[1],s["image_path"])
                except HTTPException:pass
        content=build_pdf(diary,scans,photos)
        return Response(content,media_type="application/pdf",headers={"Content-Disposition":f'attachment; filename="kulitku-{id}.pdf"',"Cache-Control":"no-store"})
    return app
app=create_app()
