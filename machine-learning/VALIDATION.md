# Validasi paket

Tanggal: 5 November 2026. Pengujian di lingkungan Linux CPU Python 3.12, bukan sesi Colab pengguna.

Lulus:
- Struktur notebook nbformat dan sintaks seluruh modul.
- EDA sebelum/sesudah cleaning dengan gambar sintetis; tabel/grafik ditulis ke folder hasil.
- Tiga arsitektur timm asli dengan bobot acak (pretrained=False khusus pengujian).
- Sembilan kombinasi classifier, lima fold, scaler di dalam training fold.
- Resume fitur selesai dan CV kandidat/fold; konfigurasi run berbeda ditolak.
- Pemilihan dari kualitas CV dan benchmark native CPU, lalu evaluasi test internal fixture.
- Ekspor dan parity fitur untuk seluruh tiga CNN dengan pipeline LR.
- Konversi dan kesetaraan label untuk seluruh sembilan pipeline sklearn (LR/SVM/RF pada tiap fitur).
- Predictor memuat dan memakai label, preprocessing, checksum dan kedua ONNX.
- Mode verified_groups menjaga kelompok augmentasi, original-only validation, dan menolak exact duplicate berlabel konflik.
- Grafik CV dan benchmark diperiksa visual; label diperbaiki agar terbaca.

Belum dilakukan:
- Training 30.684/30.699 gambar pengguna; jumlah aktual mengikuti data yang dibaca.
- Download dan pengujian weights ImageNet sebenarnya; ID ketiga pretrained weights diverifikasi terdaftar di timm 1.0.22.
- Eksekusi notebook pada akun Colab/Drive pengguna atau GPU T4.
- Validasi medis atau generalisasi antar pasien.

Fixture memakai pola warna sintetis dan bobot acak, jadi skor pengujian tidak merupakan hasil akurasi penelitian. Perubahan terakhir hanya warna teks heatmap dan legenda benchmark, lalu generator visual diuji ulang.

Versi pengujian:

```json
{
  "torch": "2.15.0+cpu",
  "torchvision": "0.29.2+cpu",
  "timm": "1.0.22",
  "numpy": "2.3.5",
  "pandas": "2.2.3",
  "Pillow": "12.0.0",
  "scikit-learn": "1.7.2",
  "onnx": "1.19.1",
  "onnxruntime": "1.23.2",
  "skl2onnx": "1.20.0"
}
```
