# Flutter Android

Source aplikasi terdapat di lib/. Dari root repo jalankan `python scripts/bootstrap_mobile.py` untuk membuat platform Android sesuai Flutter SDK terpasang. Lalu cd mobile, flutter run dengan dart-defines yang dicontohkan pada README utama.

LOCAL_DEMO=true hanya melewati login; klasifikasi tetap ditentukan mode/bundle API. Untuk fitur Diary/History persisten, gunakan mode normal dengan Supabase. Saat pemotretan dari Diary, hasil baru ditautkan setelah scan tersimpan. Mengeluarkan tautan tidak menghapus scan. Menghapus scan dari History menghapus tautannya sesuai relasi database.

PDF dibuat backend, preview dibuka oleh viewer PDF perangkat melalui open_filex; share_plus membuka Android share sheet. Pilih WhatsApp sendiri, tidak ada pengiriman otomatis. Data foto disimpan default aktif, dapat dimatikan lewat checkbox. Belum ada APK atau pengujian perangkat nyata dalam source package ini.
