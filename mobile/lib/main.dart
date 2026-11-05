import 'dart:async';
import 'core/api.dart';
import 'features/diary_page.dart';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

const localDemo = bool.fromEnvironment('LOCAL_DEMO', defaultValue: false);
const apiUrl = String.fromEnvironment(
  'API_URL',
  defaultValue: 'http://10.0.2.2:8000',
);
const supabaseUrl = String.fromEnvironment('SUPABASE_URL');
const supabaseKey = String.fromEnvironment('SUPABASE_ANON_KEY');

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  if (!localDemo) {
    if (supabaseUrl.isEmpty || supabaseKey.isEmpty) {
      runApp(
        const MaterialApp(
          home: Scaffold(
            body: Center(
              child: Text(
                'Isi SUPABASE_URL dan SUPABASE_ANON_KEY melalui --dart-define, atau gunakan LOCAL_DEMO=true.',
              ),
            ),
          ),
        ),
      );
      return;
    }
    await Supabase.initialize(url: supabaseUrl, anonKey: supabaseKey);
  }
  runApp(const KulitKuApp());
}

class KulitKuApp extends StatelessWidget {
  const KulitKuApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'KulitKu',
    theme: ThemeData(
      colorSchemeSeed: const Color(0xff196f65),
      useMaterial3: true,
    ),
    home: const HomePage(),
  );
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final email = TextEditingController();
  final password = TextEditingController();
  final picker = ImagePicker();
  StreamSubscription<AuthState>? authSubscription;
  XFile? photo;
  Map<String, dynamic>? result;
  List<dynamic> scans = [];
  bool busy = false;
  bool saveImage = true;
  String? targetDiaryId;
  Api get api => Api(apiUrl, headers);
  String? notice;
  bool get signedIn =>
      localDemo || Supabase.instance.client.auth.currentSession != null;

  @override
  void initState() {
    super.initState();
    if (!localDemo) {
      authSubscription = Supabase.instance.client.auth.onAuthStateChange.listen(
        (event) {
          if (mounted)
            setState(() {
              scans = [];
              result = null;
            });
        },
      );
    }
    recoverPhoto();
  }

  Future<void> recoverPhoto() async {
    try {
      final lost = await picker.retrieveLostData();
      if (!mounted || lost.isEmpty) return;
      if (lost.files?.isNotEmpty ?? false)
        setState(() => photo = lost.files!.first);
    } catch (e) {
      if (mounted) setState(() => notice = 'Tidak dapat memulihkan foto: $e');
    }
  }

  @override
  void dispose() {
    authSubscription?.cancel();
    email.dispose();
    password.dispose();
    super.dispose();
  }

  Future<void> action(Future<void> Function() fn) async {
    if (busy) return;
    setState(() {
      busy = true;
      notice = null;
    });
    try {
      await fn();
    } catch (e) {
      if (mounted) setState(() => notice = e.toString());
    } finally {
      if (mounted) setState(() => busy = false);
    }
  }

  Future<Map<String, String>> headers() async {
    if (localDemo) return {};
    final auth = Supabase.instance.client.auth;
    var session = auth.currentSession;
    if (session == null) throw Exception('Silakan login kembali.');
    if (session.isExpired) session = (await auth.refreshSession()).session;
    if (session == null) throw Exception('Sesi berakhir.');
    return {'Authorization': 'Bearer ${session.accessToken}'};
  }

  dynamic decodeResponse(http.Response response) {
    dynamic body;
    try {
      body = jsonDecode(response.body);
    } catch (_) {
      throw Exception('Respons server tidak valid (${response.statusCode}).');
    }
    if (response.statusCode >= 400)
      throw Exception(
        'API ${response.statusCode}: ${body is Map ? body['detail'] : body}',
      );
    return body;
  }

  Future<void> pick(ImageSource source) async {
    final file = await picker.pickImage(
      source: source,
      maxWidth: 2048,
      imageQuality: 90,
    );
    if (mounted && file != null)
      setState(() {
        photo = file;
        result = null;
      });
  }

  Future<void> scan() async {
    final selected = photo;
    if (selected == null) return;
    final bytes = await selected.readAsBytes();
    if (bytes.length > 8 * 1024 * 1024)
      throw Exception('Ukuran foto melebihi 8 MB.');
    final request = http.MultipartRequest(
      'POST',
      Uri.parse('$apiUrl/v1/scans'),
    );
    request.headers.addAll(await headers());
    request.fields['save_image'] = saveImage.toString();
    request.files.add(
      http.MultipartFile.fromBytes('file', bytes, filename: selected.name),
    );
    final streamed = await request.send().timeout(const Duration(seconds: 60));
    final response = await http.Response.fromStream(streamed)
        .timeout(const Duration(seconds: 30));
    final body = Map<String, dynamic>.from(decodeResponse(response) as Map);
    if (mounted) setState(() => result = body);
    if (targetDiaryId != null && body['saved'] == true) {
      await api.call('POST', '/v1/diaries/$targetDiaryId/scans', {'scan_id': body['scan_id']});
      if (mounted) setState(() { notice = 'Scan ditambahkan ke diary.'; targetDiaryId = null; });
    }
  }

  Future<void> history() async {
    final response = await http
        .get(Uri.parse('$apiUrl/v1/scans'), headers: await headers())
        .timeout(const Duration(seconds: 30));
    final items = decodeResponse(response) as List;
    if (mounted) setState(() => scans = items);
  }

  Future<void> deleteScan(String id) async {
    final response = await http
        .delete(Uri.parse('$apiUrl/v1/scans/$id'), headers: await headers())
        .timeout(const Duration(seconds: 30));
    decodeResponse(response);
    if (mounted) setState(() => scans.removeWhere((item) => item['id'] == id));
  }

  Widget loginForm() => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      const Text(
        'Masuk untuk menyimpan riwayat milikmu.',
        style: TextStyle(fontSize: 20),
      ),
      TextField(
        controller: email,
        keyboardType: TextInputType.emailAddress,
        decoration: const InputDecoration(labelText: 'Email'),
      ),
      TextField(
        controller: password,
        obscureText: true,
        decoration: const InputDecoration(labelText: 'Password'),
      ),
      const SizedBox(height: 16),
      FilledButton(
        onPressed: busy
            ? null
            : () => action(() async {
                await Supabase.instance.client.auth.signInWithPassword(
                  email: email.text.trim(),
                  password: password.text,
                );
                password.clear();
              }),
        child: const Text('Masuk'),
      ),
      OutlinedButton(
        onPressed: busy
            ? null
            : () => action(() async {
                await Supabase.instance.client.auth.signUp(
                  email: email.text.trim(),
                  password: password.text,
                );
                if (mounted)
                  setState(
                    () => notice =
                        'Periksa email untuk konfirmasi jika diaktifkan.',
                  );
              }),
        child: const Text('Daftar akun'),
      ),
    ],
  );

  Future<void> addToDiary(String scanId) async {
    final items = await api.call('GET', '/v1/diaries') as List;
    if (!mounted) return;
    final chosen = await showDialog<String>(context: context, builder: (ctx) => SimpleDialog(
      title: const Text('Pilih diary'), children: [
        for (final diary in items) SimpleDialogOption(onPressed: () => Navigator.pop(ctx, '${diary['id']}'), child: Text('${diary['title']}')),
        SimpleDialogOption(onPressed: () => Navigator.pop(ctx, '__new__'), child: const Text('+ Buat diary baru')),
      ]));
    if (chosen == null) return;
    String diaryId = chosen;
    if (chosen == '__new__') {
      final title = TextEditingController();
      if (!mounted) return;
      final accepted = await showDialog<bool>(context: context, builder: (ctx) => AlertDialog(
        title: const Text('Diary baru'), content: TextField(controller: title, maxLength: 150), actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Batal')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Buat'))]));
      final name = title.text.trim(); title.dispose();
      if (accepted != true || name.isEmpty) return;
      final created = await api.call('POST', '/v1/diaries', {'title': name, 'notes': ''});
      diaryId = '${created['id']}';
    }
    await api.call('POST', '/v1/diaries/$diaryId/scans', {'scan_id': scanId});
    if (mounted) setState(() => notice = 'Scan ditambahkan ke diary.');
  }

  Future<void> openDiaries() async {
    await Navigator.push(context, MaterialPageRoute<void>(builder: (_) => DiaryPage(api: api, onScan: (id) {
      Navigator.of(context).popUntil((route) => route.isFirst);
      if (mounted) setState(() => targetDiaryId = id);
      action(() => pick(ImageSource.camera));
    })));
  }

  Widget resultCard() {
    final value = result!;
    final label = value['status'] == 'classified'
        ? value['predicted_class']
        : {
                'uncertain': 'Hasil ditahan',
                'retake_required': 'Ambil ulang foto',
                'demo_only': 'Demo teknis',
              }[value['status']] ??
              'Status tidak dikenal';
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('$label', style: Theme.of(context).textTheme.headlineSmall),
            Text('${value['message']}'),
            if (value['status'] == 'classified' &&
                value['probabilities'] is Map) ...[
              const SizedBox(height: 8),
              const Text(
                'Skor classifier belum dikalibrasi; bukan probabilitas diagnosis klinis.',
              ),
              for (final entry in (value['probabilities'] as Map).entries)
                Text(
                  '${entry.key}: ${(100 * (entry.value as num)).toStringAsFixed(1)}%',
                ),
            ],
            if ((value['quality_flags'] as List?)?.isNotEmpty ?? false)
              Text('Kualitas: ${(value['quality_flags'] as List).join(', ')}'),
            Text(
              'Model: ${value['model_version']} • ${value['inference_ms']} ms',
            ),
            Text(
              value['saved'] == true
                  ? 'Riwayat tersimpan.'
                  : 'Riwayat tidak disimpan.',
            ),
            if (value['saved'] == true)
            OutlinedButton(onPressed: busy ? null : () => action(() => addToDiary('${value['scan_id']}')), child: const Text('Tambahkan ke Diary')),
        ],
        ),
      ),
    );
  }

  Widget scanner() => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      if (targetDiaryId != null) ...[const Text('Scan berikutnya akan ditambahkan ke diary yang dipilih.'), TextButton(onPressed: busy ? null : () => setState(() => targetDiaryId = null), child: const Text('Batalkan tujuan diary'))],
      if (!localDemo) OutlinedButton(onPressed: busy ? null : openDiaries, child: const Text('Buka Diary / MyHistory')),

      Text(
        localDemo ? 'Mode lokal / uji coba' : 'Scan kulit',
        style: Theme.of(context).textTheme.headlineMedium,
      ),
      const Text(
        'Ambil satu area kulit yang jelas, cukup cahaya, tanpa filter. Hindari wajah atau identitas lain bila tidak diperlukan.',
      ),
      const SizedBox(height: 12),
      if (photo != null)
        ClipRRect(
          borderRadius: BorderRadius.circular(16),
          child: Image.file(
            File(photo!.path),
            height: 260,
            fit: BoxFit.contain,
          ),
        ),
      Wrap(
        spacing: 8,
        children: [
          OutlinedButton.icon(
            onPressed: busy
                ? null
                : () => action(() => pick(ImageSource.camera)),
            icon: const Icon(Icons.camera_alt),
            label: const Text('Kamera'),
          ),
          OutlinedButton.icon(
            onPressed: busy
                ? null
                : () => action(() => pick(ImageSource.gallery)),
            icon: const Icon(Icons.photo),
            label: const Text('Galeri'),
          ),
        ],
      ),
      if (!localDemo)
        CheckboxListTile(
          value: saveImage,
          onChanged: busy
              ? null
              : (v) => setState(() => saveImage = v ?? false),
          title: const Text('Simpan foto ke riwayat privat'),
          subtitle: const Text(
            'Tanpa centang, foto hanya diproses untuk scan. Tidak digunakan untuk training.',
          ),
        ),
      FilledButton.icon(
        onPressed: busy || photo == null ? null : () => action(scan),
        icon: const Icon(Icons.search),
        label: const Text('Analisis foto'),
      ),
      if (result != null) resultCard(),
      const SizedBox(height: 16),
      OutlinedButton(
        onPressed: busy ? null : () => action(history),
        child: const Text('Muat riwayat'),
      ),
      for (final item in scans)
        Card(
          child: ListTile(
            title: Text('${item['predicted_class'] ?? item['status']}'),
            onTap: localDemo || busy ? null : () => action(() => addToDiary('${item['id']}')),
            subtitle: Text('${item['created_at']}'),
            trailing: IconButton(
              tooltip: 'Hapus scan',
              onPressed: busy
                  ? null
                  : () => action(() => deleteScan('${item['id']}')),
              icon: const Icon(Icons.delete_outline),
            ),
          ),
        ),
    ],
  );

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(
      title: const Text('KulitKu Lab'),
      actions: [
        if (!localDemo && signedIn)
          IconButton(
            tooltip: 'Keluar',
            onPressed: busy
                ? null
                : () => action(() async {
                    await Supabase.instance.client.auth.signOut();
                    if (mounted) {
                      setState(() {
                        photo = null;
                        result = null;
                        scans = [];
                        targetDiaryId = null;
                      });
                    }
                  }),
            icon: const Icon(Icons.logout),
          ),
      ],
    ),
    body: Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 640),
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            const Card(
              child: Padding(
                padding: EdgeInsets.all(12),
                child: Text(
                  'Prototipe penelitian. Hanya mendukung eczema, herpes zoster, ringworm, dan kulit normal. Hasil bukan diagnosis; kondisi lain dapat salah diklasifikasikan.',
                ),
              ),
            ),
            if (busy) const LinearProgressIndicator(),
            if (notice != null)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 12),
                child: Text(notice!, style: const TextStyle(color: Colors.red)),
              ),
            signedIn ? scanner() : loginForm(),
          ],
        ),
      ),
    ),
  );
}
