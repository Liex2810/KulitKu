# FastAPI
Dari root: `pip install -e ".[api]"` lalu `python -m uvicorn app.main:app --env-file backend/.env --host 127.0.0.1 --port 8000`.
Konfigurasi: .env.example. Swagger: /docs. Bundle terlatih di artifacts/model pada root repo. Mode demo tidak mengeluarkan label medis.

Endpoint: POST/GET /v1/scans; GET/DELETE /v1/scans/{id}; GET/POST /v1/diaries; GET/PATCH/DELETE /v1/diaries/{id}; POST /v1/diaries/{id}/scans; DELETE /v1/diaries/{id}/scans/{scan_id}; GET /v1/diaries/{id}/pdf. PDF optional query date_from/date_to menggunakan tanggal UTC scan. Keduanya YYYY-MM-DD. Filter membatasi linked scans, bukan catatan diary yang merupakan satu teks. API mengembalikan PDF saat diminta, tidak menyimpannya permanen.

Prototype membatasi daftar history/diary ke 100 terbaru; pagination belum diimplementasikan. PDF menyertakan semua linked scans yang dikembalikan query (perhatikan batas baris REST Supabase untuk data besar). Preview foto history di UI belum disediakan; foto muncul pada PDF bila disimpan.

Sebelum deploy publik: gunakan HTTPS, reverse proxy dengan body limit/rate limit, secret management, observability tanpa token/foto, uji RLS dua akun. Compose mengikat port ke localhost. Jangan gunakan mode local di internet.

Operasi Storage dan PostgreSQL bukan transaksi atomik. Upload metadata gagal memicu cleanup foto, tetapi kegagalan cleanup memerlukan rekonsiliasi. Menghapus akun melalui Auth perlu cleanup Storage tersendiri. Tidak ada endpoint penghapusan akun pada prototype ini.
