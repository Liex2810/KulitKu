# Supabase

Jalankan migrations/001_init.sql sekali pada project baru. Tabel scans, diaries, diary_scans; bucket scan-images privat. User JWT digunakan untuk seluruh akses; tidak ada service-role key.

Uji dua akun sebelum memakai data pribadi:
1. A membuat scan dan diary, kemudian menautkan keduanya.
2. A dapat membaca/edit diary, mengekspor PDF, melepas tautan, dan menghapus diary tanpa menghapus scan.
3. B tidak dapat membaca/mengubah/menghapus ID diary/scan A (API 404 atau RLS deny).
4. B tidak dapat menautkan scan A ke diary B (FK user_id gabungan/RLS menolak).
5. Tanpa JWT, endpoint privat ditolak.
6. Upload foto berada pada user_uuid/scan_uuid.jpg dan bukan URL publik.
7. Menghapus scan menghapus link-nya; menghapus diary hanya menghapus link, bukan scan/foto.
8. save_image=false tidak menyimpan foto; PDF menampilkan keterangan foto tidak tersedia.

Database pengguna adalah Supabase. Drive hanya untuk data penelitian dan hasil ML. SQL source bukan backup database.
