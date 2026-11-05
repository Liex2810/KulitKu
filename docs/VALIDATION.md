# Validasi source package

Diperiksa 5 November 2026. Source prototype, bukan aplikasi klinis siap produksi.

## Dijalankan dan lulus

- 16 tes Python: audit metadata wajib, penolakan original unknown, exact duplicate lintas kelompok, grouped holdout dan grouped folds, originals-only validation/test, stabilitas urutan manifest, pipeline scaler, API demo tanpa prediksi medis, gambar invalid/oversize, auth wajib, isolasi identitas saat query, pembuatan PDF dengan teks/gambar, pemeriksaan kedua pemilik saat link, unlink tanpa menghapus scan, dan filter tanggal PDF.
- Smoke end-to-end pada 56 gambar sintetis: inventory, reviewed lineage sintetis, grouped split, ekstraksi ketiga CNN dengan RANDOM weights (tidak mengunduh bobot), 9 kombinasi classifier dengan grouped 2-fold CV dan grid kecil, resume benchmark, pemilihan winner, final test, ekspor ONNX, numerical parity pada tensor sintetis dan development images, serta loading bundle oleh API engine. Bundle synthetic selalu menahan prediksi medis.
- Semua modul Python dan sel kode Colab diparse dengan ast.
- Arsip ZIP diperiksa integritas dan daftar file; tidak berisi dataset/bobot/credential.

Smoke test hanya memverifikasi jalur teknis, bukan akurasi medis atau mutu pretrained features. Konfigurasi nyata memakai pretrained weights dan default 5-fold; konfigurasi smoke diperkecil agar cepat.

## Belum dijalankan / perlu akun dan perangkat pengguna

- Training pada data medis aktual, audit lineage gambar augmentasi, validasi label, dan pengujian smartphone independen.
- Eksekusi pada Colab/Drive pengguna dan pemulihan sesi cloud nyata.
- SQL/RLS pada Supabase sungguhan dan uji isolasi dua akun end-to-end. Tes saat ini memakai mock untuk service/database; ikuti supabase/README.md untuk pemeriksaan nyata.
- Flutter analyze/test/build APK dan kamera/share/PDF viewer di Android nyata belum dijalankan untuk paket ini. SDK/platform disiapkan melalui scripts/bootstrap_mobile.py pada komputer pengguna. CI Flutter disediakan, tetapi belum ada hasil CI sebelum repo diunggah.
- Docker image build, deployment HTTPS, beban banyak pengguna, dan pengukuran latency jaringan.

Versi dependency pengujian dicatat pada validation_environment.json. Untuk artefak sklearn, backend mensyaratkan versi yang sama dengan training. Tidak ada hasil akurasi medis yang diklaim atau model hasil smoke yang dibundel.
