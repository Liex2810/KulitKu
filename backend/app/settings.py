from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import model_validator
class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=".env",extra="ignore")
    model_dir:str="artifacts/model"
    demo_mode:bool=False
    auth_mode:str="supabase"
    supabase_url:str=""
    supabase_anon_key:str=""
    persist_history:bool=True
    max_upload_bytes:int=8*1024*1024
    @model_validator(mode="after")
    def check(self):
        if self.auth_mode not in {"local","supabase"}:raise ValueError("Unknown auth mode")
        if self.auth_mode=="local" and self.persist_history:raise ValueError("Local mode cannot persist history")
        if self.auth_mode=="supabase" and not(self.supabase_url.startswith("https://") and self.supabase_anon_key):raise ValueError("Configure Supabase URL/key")
        return self
