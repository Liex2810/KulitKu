# Protokol eksperimen

1. Definisi kelas dan sumber/lisensi.
2. Setup versi library, seed dan path.
3. Inventory/EDA: jumlah, ukuran, contact review manual.
4. Audit metadata pasien/gambar asal dan seluruh turunan augmentasi.
5. Cleaning: include=0 untuk eksklusi beralasan; exact duplicate dihapus dengan original diprioritaskan.
6. EDA ulang.
7. Grouped holdout, kira-kira 80/20; group tidak beririsan. Derivatif augmentasi test dikarantina.
8. Frozen manifest dan 5 grouped folds pada development. Per fold: train originals+augmentasi kelompok train, validation originals kelompok validation.
9. Preprocessing: RGB, orientation correction, fixed stretch 224x224, interpolation/mean/std diambil dari konfigurasi pretrained timm. Fixed stretch dipilih untuk kontrol eksperimen dan parity backend; tidak mengklaim persis preprocessing training ImageNet asli.
10. Frozen pretrained CNN headless dengan average pooling, eval mode. Simpan state dict untuk mereproduksi ekstraktor.
11. Features sekali per model dan snapshot; matriks N x D serta manifest/hash/spec.
12. Classifier pipelines: StandardScaler+LR, StandardScaler+SVC, RF. Tidak ada PCA/SMOTE secara default.
13. GridSearchCV dengan custom indeks grouped-fold dan scorer macro-F1. Fit scaler di setiap training fold. Refit pada seluruh development oleh search.
14. Pencatatan per kombinasi: cv_results.csv, summary.json, pipeline.joblib. Rerun melewati kombinasi selesai dengan provenance identik.
15. Leaderboard dan winner berdasarkan CV mean, lalu std, lalu nama. Tuning score bersifat development; bukan estimasi independen final. Nested CV bukan bagian implementasi ini.
16. Profil CPU opsional sembilan kombinasi pada development originals. Termasuk preprocess+CNN+scaler+classifier; tidak termasuk disk decode, jaringan dan storage.
17. Final test hanya winner; output tidak ditimpa. Metrik per kelas, macro-F1, accuracy, balanced accuracy, confusion matrix, false Normal dan grouped bootstrap CI. Jangan memilih ulang pemenang dari test.
18. Evaluasi foto smartphone independen memerlukan dataset dan verifikasi label tambahan; belum otomatis tersedia. Tidak ada klaim deteksi OOD atau validasi klinis.
19. Analisis dan draft analysis.md; interpretasi akhir harus ditulis manusia berdasarkan hasil dan inspeksi error.
20. ONNX extractor + sklearn Pipeline. Periksa checker, numerical parity dan class agreement pada synthetic tensors dan validation originals.
21. Integrasi API, UI, RLS dan PDF. Uji teknis dan autentikasi dua akun.

## Audit metadata

CSV: path,label,target,sha256,phash,width,height,source,group_id,is_original,include, audit_note (opsional).

Path relatif terhadap root dataset. group_id harus global untuk pasien yang sama jika diketahui; jika tidak ada gunakan original-image ID dan nyatakan keterbatasan patient independence. include harus 0/1. is_original harus 0/1 yang telah ditinjau; unknown menghentikan split. Setidaknya satu original per kelompok. phash sama hanya kandidat: pHash tidak membuktikan lineage/pasien dan tidak tahan seluruh transformasi. Pemeriksaan near duplicate/manual tetap wajib.

## Keterbatasan statistik

Jumlah file augmentasi bukan jumlah sampel independen. Balanced file counts tidak menjamin representasi pasien seimbang. CV std bukan confidence interval populasi. Bootstrap CI berdasarkan kelompok test, dengan 300 pengulangan awal; data sedikit tetap menghasilkan ketidakpastian besar. Dataset bersumber dari Kaggle tidak otomatis mewakili foto smartphone nyata.

## Reproduksibilitas

Simpan manifest, config, code commit, dataset version, seed, library versions dan hasil setiap run. Feature extraction merekam torch/timm dan hash bobot. Root config menentukan grid dan folds. Gunakan versi folder baru untuk perubahan. Tidak ada resume CV di tengah satu kombinasi; ekstraksi resume per batch. External test tetap harus diisolasi dan tidak dipakai untuk tuning.
