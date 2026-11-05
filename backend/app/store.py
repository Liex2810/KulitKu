import httpx
from fastapi import HTTPException
class Store:
    def __init__(self,settings):self.s=settings
    def request(self,method,path,token,**kwargs):
        headers={"apikey":self.s.supabase_anon_key,"Authorization":f"Bearer {token}",**kwargs.pop("headers",{})}
        try:
            r=httpx.request(method,self.s.supabase_url.rstrip("/")+path,headers=headers,timeout=30,**kwargs)
            if r.status_code in {401,403}:raise HTTPException(401,"Invalid token or access denied")
            r.raise_for_status();return r
        except httpx.HTTPError as e:raise HTTPException(502,"Supabase request failed") from e
    def user(self,token):
        try:return self.request("GET","/auth/v1/user",token).json()["id"]
        except (KeyError,ValueError) as e:raise HTTPException(401,"Invalid identity") from e
    def rows(self,token,table,user,**filters):
        return self.request("GET",f"/rest/v1/{table}",token,params={"user_id":f"eq.{user}","select":"*",**filters}).json()
    def get(self,token,table,user,id):
        rows=self.rows(token,table,user,id=f"eq.{id}")
        if not rows:raise HTTPException(404,"Item not found")
        return rows[0]
    def insert(self,token,table,row):
        self.request("POST",f"/rest/v1/{table}",token,json=row,headers={"Prefer":"return=minimal"})
    def update(self,token,table,user,id,values):
        self.get(token,table,user,id)
        self.request("PATCH",f"/rest/v1/{table}",token,params={"id":f"eq.{id}","user_id":f"eq.{user}"},json=values)
    def delete(self,token,table,user,id):
        row=self.get(token,table,user,id)
        if table=="scans" and row.get("image_path"):self.remove_photo(token,row["image_path"])
        self.request("DELETE",f"/rest/v1/{table}",token,params={"id":f"eq.{id}","user_id":f"eq.{user}"})
    def upload(self,token,path,content):
        self.request("POST",f"/storage/v1/object/scan-images/{path}",token,content=content,headers={"Content-Type":"image/jpeg"})
    def photo(self,token,path):return self.request("GET",f"/storage/v1/object/authenticated/scan-images/{path}",token).content
    def remove_photo(self,token,path):self.request("DELETE","/storage/v1/object/scan-images",token,json={"prefixes":[path]})
    def link(self,token,user,diary,scan):
        self.get(token,"diaries",user,diary);self.get(token,"scans",user,scan)
        self.request("POST","/rest/v1/diary_scans",token,json={"user_id":user,"diary_id":diary,"scan_id":scan},headers={"Prefer":"resolution=ignore-duplicates"})
    def unlink(self,token,user,diary,scan):
        self.get(token,"diaries",user,diary)
        self.request("DELETE","/rest/v1/diary_scans",token,params={"user_id":f"eq.{user}","diary_id":f"eq.{diary}","scan_id":f"eq.{scan}"})
    def linked(self,token,user,diary):
        self.get(token,"diaries",user,diary)
        rows=self.request("GET","/rest/v1/diary_scans",token,params={"user_id":f"eq.{user}","diary_id":f"eq.{diary}","select":"scan_id,scans(*)"}).json()
        return sorted([r["scans"] for r in rows if r.get("scans")],key=lambda r:r["created_at"])
