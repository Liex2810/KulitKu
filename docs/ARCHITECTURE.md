# Arsitektur

| Sistem | Tanggung jawab |
|---|---|
| GitHub monorepo | Source dan konfigurasi seluruh komponen |
| Google Drive | Dataset, manifest, fitur, hasil eksperimen, model dan export |
| Google Colab | Ekstraksi GPU, tuning classifier CPU, evaluasi dan ekspor |
| Flutter | Kamera, Auth, hasil, history/diary dan PDF sharing |
| FastAPI | Validasi pengguna/foto, ONNX+sklearn inferensi, diary dan PDF |
| Supabase Auth/PostgreSQL/Storage | Identitas, metadata, relasi dan foto privat |

Model diambil dari export dan dipasang saat deployment; backend tidak mengunduh Drive setiap request. Backbone ONNX dimuat per worker; sklearn pipeline mencakup scaler+classifier. Paket deployment harus berasal dari run yang sama, versi sklearn harus cocok. Local API mode tidak memerlukan Supabase dan tidak menyimpan data. Supabase mode memakai public key + caller JWT; key service-role tidak diperlukan.

ML pipeline berbeda dari versi prototype sebelumnya yang langsung fine-tuning CNN. Versi ini membekukan CNN sebagai ekstraktor dan melakukan grouped CV pada classifier klasik. Tidak memasukkan calibration/selective prediction sebagai eksperimen utama.

Scope UI: autentikasi dasar, scan/history, diary CRUD, multi-select history, scan dari diary, hasil scan ke diary, PDF backend, preview eksternal dan share sheet. Ini starter yang perlu uji integrasi nyata; bukan aplikasi klinis siap produksi.
