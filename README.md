# KulitKu Monorepo

## Alur branch dan pull request

Buat branch fitur dari `develop` yang terbaru. Gunakan ID kartu Trello pada nama branch, misalnya `feature/US-01-registrasi`.

```bash
git switch develop
git pull --ff-only origin develop
git switch -c feature/US-01-registrasi
```

Setelah commit, jalankan `git push -u origin feature/US-01-registrasi`. Buat pull request dengan **base** `develop` dan **compare** branch fitur. Cantumkan ID atau tautan kartu Trello di pull request.

Flutter Android + FastAPI + Supabase, dengan eksperimen **3 frozen pretrained CNN × 3 classic ML classifiers**. Kode di GitHub, dataset/artefak di Google Drive, komputasi di Google Colab.

**Paket source prototype yang dapat dijalankan; tidak menyertakan dataset, model medis terlatih, credential, atau APK.** Mode demo hanya memeriksa koneksi dan tidak mengeluarkan prediksi medis. Validasi yang benar-benar telah dilakukan dicatat di `docs/VALIDATION.md`.

## Komponen

| Folder | Isi |
|---|---|
| mobile | Flutter: login/daftar, kamera/galeri, scan, history, diary, PDF preview/berbagi |
| backend | FastAPI: inferensi ONNX+sklearn, autentikasi, scan, diary, PDF on-demand |
| machine-learning | Audit data, grouped split, fitur 3 CNN, 9 kombinasi grouped CV, test final, ekspor |
| shared | Preprocessing gambar yang identik untuk ekstraksi dan inferensi |
| supabase/migrations | PostgreSQL, relasi scan-diary, Storage privat, RLS |
| scripts | Bootstrap platform Android, publikasi GitHub, synthetic end-to-end smoke |
| docs | Setup, protokol, kontrak API dan batas pengujian |
| .github/workflows | Tes Python dan pemeriksaan/build Flutter |

## Mulai dari mana?

1. **Coba koneksi API** tanpa dataset (bagian A).
2. **Coba Flutter** dengan backend demo (bagian B).
3. **Publikasikan repo** (bagian C).
4. **Siapkan Supabase** (bagian D).
5. **Siapkan data di Drive dan jalankan Colab** (bagian E).
6. **Pasang bundle terlatih** untuk inferensi nyata (bagian F).

## A. API demo lokal — tanpa training atau Supabase

Python 3.10+; disarankan 3.11. Jalankan dari root hasil ekstraksi:

```bash
python -m venv .venv
```
Aktivasi Windows PowerShell: `.venv\Scripts\Activate.ps1`. macOS/Linux: `source .venv/bin/activate`.

```bash
python -m pip install -e ".[api,dev]"
```

Buat `backend/.env`:
```dotenv
DEMO_MODE=true
AUTH_MODE=local
PERSIST_HISTORY=false
```

Jalankan:
```bash
python -m uvicorn app.main:app --env-file backend/.env --host 127.0.0.1 --port 8000
```
Buka http://127.0.0.1:8000/docs → POST /v1/scans → unggah foto. Hasil harus `demo_only`, tanpa label medis. `/health` menampilkan mode aktif. Di mode local, history/diary tidak disimpan dan endpoint mutasi diary ditolak. Jangan mengekspos AUTH_MODE=local ke internet.

## B. Flutter Android

Install Flutter stable dan Android SDK di komputer, lalu periksa `flutter doctor`.

```bash
python scripts/bootstrap_mobile.py
cd mobile
flutter run --dart-define=LOCAL_DEMO=true --dart-define=API_URL=http://10.0.2.2:8000
```

Bootstrap membuat platform Android menggunakan SDK terpasang tanpa menimpa lib/. Source Android belum dibundel sebagai scaffold versi SDK tertentu. HTTP lokal hanya diizinkan pada debug; release gunakan HTTPS.

- Emulator Android: API host komputer adalah `10.0.2.2`.
- HP fisik via USB: jalankan `adb reverse tcp:8000 tcp:8000`, lalu gunakan API_URL=http://127.0.0.1:8000.
- Source ditujukan untuk Android, bukan Flutter Web (menggunakan dart:io).

Untuk mode Supabase:
```bash
flutter run --dart-define=API_URL=https://YOUR_API --dart-define=SUPABASE_URL=https://YOUR_PROJECT.supabase.co --dart-define=SUPABASE_ANON_KEY=YOUR_PUBLIC_KEY
```

UI menyediakan kamera/galeri, history dan tombol Diary. Dari hasil scan tersimpan atau item history pengguna bisa menautkan ke diary; diary dapat dibuat/diedit dan memilih beberapa scan dari history. Tombol Scan baru di diary membuka kamera dan menautkan hasil setelah tersimpan. PDF dibuat backend saat diminta, dibuka oleh PDF viewer perangkat, dan dibagikan melalui menu Android (termasuk WhatsApp jika terpasang). Pemilihan tanggal UI, ensiklopedia, chatbot dan profil lengkap belum termasuk prototype ini.

## C. Publikasikan ke GitHub

Install Git dan GitHub CLI dari https://cli.github.com. Konfigurasikan identitas Git jika belum pernah commit.

```powershell
gh auth login
powershell -ExecutionPolicy Bypass -File scripts/publish_github.ps1 -Repo Liex2810/kulitku
```
Skrip membuat **repo private** baru. Tambahkan `-Public` hanya bila ingin kode terbuka. Nama repo bisa diganti. Untuk macOS/Linux: `bash scripts/publish_github.sh Liex2810/kulitku`.

Jika repo kosong sudah dibuat melalui web:
```bash
git init -b main
git add .
git commit -m "Initialize KulitKu monorepo"
git remote add origin https://github.com/Liex2810/kulitku.git
git push -u origin main
```
GitHub berisi kode, config, notebook, SQL dan docs. Jangan commit dataset, bobot, fitur, foto pengguna, token atau .env. Masing-masing komponen tetap bisa memiliki environment dan deployment terpisah.

## D. Supabase

1. Buat project Supabase.
2. Jalankan `supabase/migrations/001_init.sql` sekali di SQL Editor pada project baru.
3. Aktifkan email/password Auth; konfirmasi email pengguna sesuai pengaturan project.
4. Salin `backend/.env.example` menjadi `backend/.env`, isi URL dan anon/publishable key.
5. Set DEMO_MODE=false setelah bundle terlatih tersedia; AUTH_MODE=supabase, PERSIST_HISTORY=true.
6. Jalankan backend memakai --env-file backend/.env seperti bagian A.

Tidak ada service-role key. Backend memverifikasi JWT melalui Supabase Auth, lalu memakai JWT pengguna yang sama untuk REST/Storage agar RLS berlaku. Foreign key gabungan memastikan scan dan diary milik pengguna yang sama. Uji dua akun sesuai `supabase/README.md` sebelum menggunakan data pribadi.

Foto scan disimpan otomatis pada mode normal bila checkbox Simpan foto aktif (default aktif); metadata scan disimpan otomatis. Jika checkbox dimatikan, foto tidak disimpan. Gambar di-encode ulang tanpa EXIF/GPS. Foto/diary aplikasi berada di Supabase, bukan Drive dataset penelitian. Data pengguna tidak otomatis digunakan untuk training.

## E. Drive + Colab

Buka `machine-learning/notebooks/KulitKu_Colab.ipynb` melalui https://colab.research.google.com (Upload notebook). Pilih runtime GPU untuk ekstraksi.

Struktur Drive:
```text
MyDrive/KulitKu/
  code/KulitKu_Monorepo.zip          # mode ZIP untuk private repo/belum terbit
  datasets/raw/
    Eczema/
    Herpes Zoster/
    Normal/
    Ringworm/
  manifests/
  features/v1/
  experiments/benchmark-v1/
  exports/skin-v1/
```

Notebook mengambil kode melalui clone GitHub publik ATAU ZIP di Drive (default). Dataset disalin ke `/content` untuk pembacaan cepat. Fitur dan kemajuan ekstraksi ditulis ke Drive setiap batch. Kombinasi classifier yang selesai disimpan dan dilewati saat rerun; pencarian yang terputus di tengah satu kombinasi harus diulang. `/content` bersifat sementara.

**Audit augmentasi wajib.** Dataset yang sudah diaugmentasi tidak boleh langsung random split. Sel inventory menghasilkan CSV; buat reviewed-v1.csv dan isi group_id (pasien/gambar asal), is_original (1/0), include (1/0). Jika metadata asal tidak diketahui, pipeline berhenti; cari dataset sebelum augmentasi atau telusuri sumbernya. Jangan mengarang group_id atau menandai seluruh gambar sebagai original.

Rancangan: grouped holdout kira-kira 80/20, lalu grouped 5-fold CV pada development. Training fold memakai original+augmentasi kelompok training. Validation dan test menggunakan originals terverifikasi. Pemenang dipilih berdasarkan mean macro-F1 CV, lalu std lebih rendah sebagai tie-break, kemudian nama kombinasi agar deterministik. GridSearchCV me-refit pemenang tiap kombinasi pada seluruh development. Test final hanya mengevaluasi kombinasi yang sudah dipilih. Lihat `docs/ML_PROTOCOL.md`.

Tiga CNN: MobileNetV4-Conv-Small, ResNet50, ConvNeXt-Tiny. Tiga classifier: LR, SVM, RF. Bobot CNN dibekukan; bukan fine-tuning. SVM tidak mengaktifkan probability=True, sehingga hasilnya berupa kelas tanpa skor probabilitas. Skor LR/RF belum dikalibrasi. Tidak ada selective prediction atau detektor OOD pada versi ini.

## F. Bundle model untuk backend

Colab menghasilkan folder `exports/skin-v1` dan ZIP bundle. Ekstrak isi ZIP ke `artifacts/model/` di root repo:
```text
artifacts/model/
  extractor.onnx
  classifier.joblib
  preprocessing.json
  labels.json
  release.json
  parity.json
  checksums.json
  model_card.md
```

Jangan mencampur file dari run berbeda. Gunakan versi scikit-learn yang sama dengan environment training (tercatat di release.json). `joblib` berbasis pickle: hanya muat bundle buatanmu sendiri dari sumber yang dipercaya. Checksum memeriksa konsistensi, bukan membuktikan sumber tepercaya.

CNN diekspor ONNX; scaler+classifier tetap satu sklearn Pipeline. Backend tidak perlu torch/timm. Model dimuat satu kali per worker, bukan setiap request. Untuk uji model terlatih tanpa Supabase, gunakan DEMO_MODE=false, AUTH_MODE=local, PERSIST_HISTORY=false.

## G. CLI eksperimen

Dari root repo:
```bash
python -m pip install -e ".[ml]"
python -m kulitku_ml.data inventory --data /path/to/raw --output /path/to/inventory.csv
```
Selesaikan review CSV lalu:
```bash
python -m kulitku_ml.data split --manifest /path/to/reviewed.csv --config machine-learning/configs/default.json --output /path/to/split-v1 --approve-audit
python -m kulitku_ml.features --data /path/to/raw --manifest /path/to/split-v1/manifest.csv --config machine-learning/configs/default.json --model mobilenetv4 --output /path/to/features/v1/mobilenetv4
```
Ulangi ekstraksi untuk `resnet50` dan `convnext_tiny`, dengan config/manifest yang sama. Kemudian:
```bash
python -m kulitku_ml.benchmark run --features /path/to/features/v1 --config machine-learning/configs/default.json --output /path/to/benchmark-v1
python -m kulitku_ml.profile --benchmark /path/to/benchmark-v1 --data /path/to/raw
python -m kulitku_ml.benchmark evaluate --benchmark /path/to/benchmark-v1 --output /path/to/final-test-v1
python -m kulitku_ml.export --benchmark /path/to/benchmark-v1 --output /path/to/export-v1 --version skin-v1 --data /path/to/raw
```

Config tidak boleh berubah ketika resume. Buat versi folder baru untuk perubahan dataset/config/seed. Grid default adalah titik awal, bukan konfigurasi optimal. CV tuning bukan pengukuran final independen. Belum ada eksperimen nested CV, validasi klinis, atau evaluasi eksternal smartphone otomatis.

## H. Pengujian

```bash
python -m pip install -e ".[ml,api,dev]"
python -m pytest -q
```
Opsional CPU-only: install torch/torchvision dahulu dari https://download.pytorch.org/whl/cpu untuk menghindari paket CUDA besar.

Smoke end-to-end tanpa dataset medis atau bobot pretrained:
```bash
python scripts/smoke_ml.py --output /path/to/new/synthetic-smoke
```
Jangan memakai hasil smoke sebagai hasil penelitian. Seluruh bundle dari random-init ditandai synthetic_demo dan backend menahan label medis.

Flutter setelah bootstrap:
```bash
cd mobile
flutter analyze
flutter test --dart-define=LOCAL_DEMO=true
flutter build apk --debug --dart-define=LOCAL_DEMO=true
```
GitHub Actions disediakan untuk pemeriksaan setelah repo dipublikasikan. Batas validasi aktual: docs/VALIDATION.md.
