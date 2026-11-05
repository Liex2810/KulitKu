-- Run ONCE on a fresh Supabase project. Source migrations are not database backups.
create table public.scans (
 id uuid primary key, user_id uuid not null references auth.users(id) on delete cascade,
 created_at timestamptz not null default now(), image_path text,
 status text not null check(status in ('classified','uncertain','retake_required')),
 predicted_class text check(predicted_class in ('Eczema','Herpes Zoster','Normal','Ringworm')),
 probabilities jsonb, quality_flags jsonb not null default '[]',
 model_version text not null, inference_ms double precision not null,
 unique(id,user_id),
 check(image_path is null or split_part(image_path,'/',1)=user_id::text)
);
create table public.diaries (
 id uuid primary key, user_id uuid not null references auth.users(id) on delete cascade,
 created_at timestamptz not null default now(), title text not null check(length(title) between 1 and 150),
 notes text not null default '' check(length(notes)<=20000), unique(id,user_id)
);
create table public.diary_scans (
 user_id uuid not null references auth.users(id) on delete cascade,
 diary_id uuid not null, scan_id uuid not null,
 primary key(diary_id,scan_id),
 foreign key(diary_id,user_id) references public.diaries(id,user_id) on delete cascade,
 foreign key(scan_id,user_id) references public.scans(id,user_id) on delete cascade
);
create index scans_user_created on public.scans(user_id,created_at desc);
create index diaries_user_created on public.diaries(user_id,created_at desc);
create index diary_scans_user on public.diary_scans(user_id);
alter table public.scans enable row level security;
alter table public.diaries enable row level security;
alter table public.diary_scans enable row level security;
revoke all on public.scans,public.diaries,public.diary_scans from anon;
grant select,insert,delete on public.scans,public.diary_scans to authenticated;
grant select,insert,update,delete on public.diaries to authenticated;
create policy scan_read on public.scans for select to authenticated using((select auth.uid())=user_id);
create policy scan_add on public.scans for insert to authenticated with check((select auth.uid())=user_id);
create policy scan_delete on public.scans for delete to authenticated using((select auth.uid())=user_id);
create policy diary_read on public.diaries for select to authenticated using((select auth.uid())=user_id);
create policy diary_add on public.diaries for insert to authenticated with check((select auth.uid())=user_id);
create policy diary_edit on public.diaries for update to authenticated using((select auth.uid())=user_id) with check((select auth.uid())=user_id);
create policy diary_delete on public.diaries for delete to authenticated using((select auth.uid())=user_id);
create policy link_read on public.diary_scans for select to authenticated using((select auth.uid())=user_id);
create policy link_add on public.diary_scans for insert to authenticated with check((select auth.uid())=user_id);
create policy link_delete on public.diary_scans for delete to authenticated using((select auth.uid())=user_id);
insert into storage.buckets(id,name,public,file_size_limit,allowed_mime_types)
 values('scan-images','scan-images',false,8388608,array['image/jpeg']) on conflict(id) do nothing;
create policy image_read on storage.objects for select to authenticated using(bucket_id='scan-images' and (storage.foldername(name))[1]=(select auth.uid())::text);
create policy image_add on storage.objects for insert to authenticated with check(bucket_id='scan-images' and (storage.foldername(name))[1]=(select auth.uid())::text);
create policy image_delete on storage.objects for delete to authenticated using(bucket_id='scan-images' and (storage.foldername(name))[1]=(select auth.uid())::text);
