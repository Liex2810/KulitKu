# KulitKu ML — cara pakai dan kontrak antarfile

## Mulai di Google Colab

1. Ekstrak ZIP paket di laptop. Buka `notebooks/kulitku_experiment.ipynb` melalui File → Upload notebook di Colab.
2. Pilih runtime GPU T4 jika tersedia. Notebook tetap bisa memakai CPU, tetapi ekstraksi lebih lambat.
3. Jalankan bagian 1. Isi REPO_URL jika sudah mengunggah kode ke GitHub publik; jika kosong, sel meminta upload ZIP paket utuh ini ke runtime. Dataset tidak di-upload melalui sel ini.
4. Jalankan instalasi dependensi. Jika Colab meminta restart, restart lalu jalankan bagian 1, lewati instalasi dengan INSTALL_DEPENDENCIES=False, dan lanjut bagian 3.
5. Mount Drive dan isi DATASET_ROOT_OVERRIDE bila lokasi bukan `/content/drive/MyDrive/KulitKu/datasets/original`. Folder tersebut langsung berisi Eczema, Herpes Zoster, Normal, Ringworm. OUTPUT_ROOT_OVERRIDE opsional.
6. Jalankan tiap bagian notebook berurutan. Boleh berhenti setelah EDA cleaning; tahap berikutnya tidak berjalan sendiri.
7. Simpan RUN_ID yang dicetak. Untuk sesi baru, siapkan kode/paket versi sama, isi RUN_ID tersebut, dan jalankan bagian yang dibutuhkan. Setelah konfigurasi, hasil `before`, `after`, `splits` dapat dimuat melalui `run.read('inventory.csv')`, `run.read('cleaned.csv')`, `run.read('splits.csv')`. Gunakan input dataset versi tetap.

Data sumber tidak dihapus. Folder hasil dibuat otomatis saat Run dibuat; subfolder CNN/classifier dibuat saat tahapnya digunakan. Grafik dan tabel langsung ditulis ke Drive, bukan menunggu akhir notebook.

## Struktur yang benar-benar dipakai

- `notebooks/kulitku_experiment.ipynb`: urutan kerja, pemanggilan modul, tampilan hasil; bukan salinan fungsi implementasi.
- `configs/experiment.json`: satu konfigurasi yang dibaca `load_config`, diteruskan ke semua modul melalui `Run`.
- `src/kulitku_ml/config.py`: validasi konfigurasi, pembuatan folder, identitas run, penulisan grafik/CSV/JSON.
- `data.py`: inventaris dan decode gambar tanpa hash di awal.
- `eda.py`: distribusi, resolusi, rasio, format/mode, galeri, histogram RGB, indikator sampel dan perbandingan sebelum–sesudah.
- `cleaning.py`: keputusan teknis dan CSV review manual, tidak menghapus gambar sumber.
- `preprocessing.py`: EXIF → alpha putih jika perlu → RGB, transformasi deterministik sesuai pretrained weights timm.
- `splitting.py`: hash exact duplicate satu kali di tahap split, deduplikasi/konflik label, holdout berkelompok, lima fold development.
- `features.py`: tiga CNN dibekukan dan mode evaluasi, ekstraksi dan cache fitur, penyimpanan state weights dan transformasi.
- `training.py`: pipeline LR/SVM/RF, pencarian grid dengan indeks CV eksplisit, progress tiap fold, refit development, prediksi validation.
- `evaluation.py`: laporan kelas, confusion matrix jumlah/normalisasi, galeri salah, heatmap, fold/error bar, test dan CI bootstrap kelompok.
- `benchmarking.py`: pengukuran semua pipeline native CPU pada sampel development, pemilihan berdasar aturan yang dikunci.
- `export.py`: CNN dan pipeline scaler/classifier → ONNX, parity sebelum release.
- `inference.py`: Predictor benar-benar membaca labels.json/preprocessing.json/release.json lalu kedua ONNX; untuk diintegrasikan ke endpoint FastAPI berikutnya.
- `requirements.txt`: dependensi yang dipasang sel instalasi. Pin komponen model/ONNX dan rentang library bawaan; `environment.json` dan `requirements_snapshot.txt` merekam versi aktual. Bukan lock universal semua image Colab.
- `pyproject.toml`: pemasangan package src dengan pip editable.
- `tests/smoke_pipeline.py`: pengujian sintetis terpisah, bukan langkah training datasetmu.

Tidak ada berkas Python kosong yang disertakan sebagai janji implementasi. Folder src adalah implementasi yang dipakai notebook.

## Konfigurasi — apa yang diatur

| Bagian JSON | Dipakai oleh |
|---|---|
| paths | sumber gambar dan folder hasil semua tahap |
| seed, classes | split, model, sampling, label metrik/inferensi |
| cleaning | data.py, cleaning.py, decoder preprocessing |
| eda | ukuran sampel dan galeri |
| split | strategi kelompok dan jumlah fold |
| extractors, features | ID timm, pretrained, batch ekstraksi |
| classifiers, training | grid hyperparameter, CPU jobs |
| selection | toleransi absolut macro F1 0.01 |
| benchmark | thread, sampel per kelas, warmup, pengulangan |
| evaluation | replikasi bootstrap final |
| export | gambar per kelas untuk parity |

`run_config.json` adalah snapshot konfigurasi efektif + fingerprint kode dan versi. Jangan edit snapshot untuk mengubah eksperimen; ubah experiment.json dan buat run baru. Path override notebook juga dicatat. Mengubah config/kode/library pada RUN_ID lama ditolak.

## Dua mode split: jangan menukar makna hasil

**Default `exploratory_hash`:** supaya pipeline bisa dicoba dengan dataset saat ini yang asli/augmentasinya tercampur. Hash byte mengelompokkan exact duplicate dan hanya satu representatif dipertahankan. Hubungan crop/rotasi/flip atau foto pasien sama belum diketahui. HASIL EKSPLORATORI tetap berisiko kebocoran augmentasi; jangan klaim validasi independen/klinis. K-fold tidak otomatis memperbaiki masalah ini. Grafik model utama dan JSON hasil diberi status eksploratori. Ini perubahan eksplisit dari protokol final yang mensyaratkan lineage terverifikasi; gunakan untuk uji coba, bukan menyatakan protokol final sudah terpenuhi.

**`verified_groups`:** untuk eksperimen setelah audit. Isi CSV dengan kolom dari configs/lineage_template.csv dan ubah split.lineage_csv ke path Drive, split.mode ke verified_groups. Setiap gambar yang dipakai memerlukan lineage_group, original_status (verified_original / verified_augmented), serta evidence yang nyata. Jangan menebak pasien atau menyimpulkan original hanya dari nama. Kelompok asal tidak melintasi development/test maupun fold. Validation/test hanya verified_original; turunan dari kelompok validation/test tidak ikut training. Jika dukungan kelompok/kelas tidak cukup, kode berhenti sebelum melaporkan skor menyesatkan. Tanpa patient ID, jangan klaim pemisahan antar pasien.

Konflik label pada file byte-identik tidak ditebak: tabel duplicate_label_conflicts.csv dibuat, lalu proses berhenti. Selesaikan melalui review cleaning dan run baru. Jika hanya ingin EDA/cleaning, berhenti sebelum bagian split; hashing belum dijalankan.

## Cleaning yang benar-benar dilakukan

Otomatis: file gagal decode, ekstensi/isi di luar JPG/JPEG/PNG/WEBP, batas 25MP, symlink, multiframe (karantina). RGB-convertible tetap diterima. Minimum 128px hanya flag secara default; exclude memerlukan alasan di config. Tidak membuang berdasarkan brightness, blur, atau token aug pada nama. Tidak ada deteksi watermark/diagnosis otomatis.

`manual_review_template.csv` dibuat di hasil/tabel. Isi salinan dengan decision keep/exclude, reason, reviewer; label_override perlu label_evidence dan label target yang valid. Isi cleaning.manual_review_csv pada konfigurasi, buat run baru. Pencocokan memeriksa image_id DAN relative_path. Daftar gambar/keputusan CSV tetap dipakai agar sembilan kombinasi memakai data sama; tidak memerlukan folder manifests terpisah.

## CNN beku dan classifier

MobileNetV4-Conv-Small (`mobilenetv4_conv_small.e2400_r224_in1k`), ResNet50 (`resnet50.a1_in1k`), ConvNeXt-Tiny (`convnext_tiny.fb_in1k`). Pretrained ImageNet, kepala classifier dilepas, global pooling, semua parameter requires_grad=False dan eval. Transformasi tiap weights diselesaikan oleh timm dan disimpan pada features/<cnn>/metadata.json. Tidak semua preset harus memakai resize/crop identik; deployment mengikuti preset yang benar-benar dipakai.

CNN tidak belajar dari label KulitKu, sehingga fitur dapat di-cache sekali. LR/SVM memakai StandardScaler yang di-fit hanya pada train fold; RF tanpa scaler. Sembilan kombinasi berbagi pembagian data sama. Test tidak dipakai scaler/tuning/pemilihan.

Default grid: 6 LR, 12 SVM, 8 RF per CNN. Total 390 fit pencarian pada 5-fold, ditambah refit dan pembuatan OOF terbaik. Ini dapat memerlukan beberapa sesi Colab gratis. Gunakan filter EXTRACT_ONLY, TRAIN_ONLY_CNN, TRAIN_ONLY_CLASSIFIER untuk menjadwalkan, bukan mengubah grid di tengah run.

Resume didukung untuk CNN yang ekstraksinya sudah selesai dan kandidat/fold classifier yang selesai. Batch ekstraksi yang belum selesai belum disimpan ke Drive dan diulang. Cache inventory/split mengasumsikan dataset tetap; jangan mengubah file sumber saat menggunakan RUN_ID sama. Setelah data/config berubah, gunakan RUN_ID baru.

## Benchmark dan aturan pemilihan

Runtime pembanding: PyTorch CPU + sklearn untuk semua kandidat, batch1, jumlah thread tetap. RF predict jobs1; CNN/scaler/classifier dan decode/preprocessing ikut timer, initial load/network tidak. Model refit development disiapkan di tahap training, lalu digunakan untuk benchmark dan final test; tidak perlu refit identik lagi.

Pilih kombinasi dengan mean macro F1 CV >= terbaik−0.01; dari kandidat pilih median latency paling kecil, lalu ukuran gabungan, lalu combo_id. Ini toleransi keputusan, bukan uji kesetaraan statistik. selection.json dibuat sebelum test; benchmark tidak mengganti pilihan yang sudah dikunci.

Benchmark native tidak menjamin peringkat tercepat ONNX sama. Ekspor dilakukan setelah test; jika tujuan diubah menjadi memilih berdasarkan ONNX, protokol harus direvisi sebelum membuka test dan semua kandidat perlu diukur pada ONNX. Waktu inferensi CPU bukan waktu respons aplikasi yang mencakup jaringan.

## Output Drive otomatis

```
KulitKu/hasil/<run_id>/
├── grafik/                 # EDA, CV, kelas, confusion matrix, benchmark
├── tabel/                  # inventaris, cleaning, split, fitur-order, CV, test, parity
├── features/<cnn>/         # features.npy, extractor.pt, metadata.json
├── models/<cnn>/<ml>/      # progress CV, classifier.joblib, cv_summary.json
├── onnx/<pilihan>/         # extractor.onnx, classifier.onnx, labels/preprocessing/release.json
├── run_config.json
├── environment.json
├── requirements_snapshot.txt
├── split_status.json
├── selection.json
└── test_metrics.json
```

Dataset tidak disalin menjadi folder cleaned; daftar kandidat pada tabel/cleaned.csv dibaca oleh tahap berikutnya. Tidak ada grafik/tabel yang hanya ditampilkan tanpa ekspor oleh fungsi tahap yang bersangkutan. Galeri menggunakan foto data, jadi jangan publikasikan sembarangan.

## ONNX dan konsumen hasil

CNN serta pipeline scaler/classifier diekspor terpisah. Transformasi JPEG→tensor berada di luar grafik ONNX; preprocessing.json dipakai Predictor untuk membuat transformasi yang sama melalui timm, sedangkan labels.json memetakan output integer. `release.json` hanya dibuat jika checker dan parity fitur/label berhasil pada sampel development. Ini uji numerik, bukan jaminan akurasi klinis.

Jika converter atau prediksi parity gagal, tidak ada release siap dipakai. Jangan menganggap file .onnx yang terbentuk sudah terverifikasi. ONNX Runtime CPU 1.23.2 dipin; versi lain harus diuji ulang. Predictor masih membutuhkan timm/torchvision untuk transformasi, walaupun forward model memakai ONNX; backend minimal tanpa torch perlu implementasi preprocessing ekuivalen dan uji parity tambahan.

## Pengujian

Dari folder machine-learning setelah dependensi terpasang:

```bash
python tests/smoke_pipeline.py
```

Pengujian memakai gambar sintetis, tiga arsitektur asli dengan bobot ACAK (tidak mengunduh pretrained), grid kecil dan lima fold. Skor fixture bukan hasil penelitian. Validasi mencakup 9 pipeline classifier, cache, isolasi split, grafik/CSV, pemilihan, test, serta ekspor/parity. Pengujian mungkin memakai beberapa ratus MB RAM/disk karena weights model asli.

## Referensi primer

- https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k
- https://huggingface.co/timm/resnet50.a1_in1k
- https://huggingface.co/timm/convnext_tiny.fb_in1k
- https://onnx.ai/sklearn-onnx/supported.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

Penggunaan akhir adalah bantuan skrining awal, bukan diagnosis. Dataset Kaggle dan anotasinya tetap harus ditinjau; normal bukan jaminan bebas seluruh penyakit.
