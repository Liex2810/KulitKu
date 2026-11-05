# Machine learning

Jalankan semua perintah dari root monorepo setelah `pip install -e ".[ml]"`. Notebook utama: notebooks/KulitKu_Colab.ipynb.

Source: src/kulitku_ml/data.py (inventory, audit gate, split), features.py (frozen timm features + batch resume), benchmark.py (grouped CV/tuning/refit/final-test), profile.py (CPU latency 9 candidates), export.py (ONNX extractor + sklearn bundle).

Lihat root README dan docs/ML_PROTOCOL.md. Dataset/fitur/model tidak disimpan di GitHub. Untuk data yang sudah diaugmentasi, metadata lineage harus direview; pipeline tidak menebak asal gambar dari nama secara otomatis.
