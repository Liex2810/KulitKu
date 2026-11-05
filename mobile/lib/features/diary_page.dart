import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:open_filex/open_filex.dart';
import '../core/api.dart';

class DiaryPage extends StatefulWidget {
  const DiaryPage({super.key, required this.api, required this.onScan});
  final Api api;
  final void Function(String) onScan;
  @override
  State<DiaryPage> createState() => _DiaryPageState();
}
class _DiaryPageState extends State<DiaryPage> {
  List<dynamic> items = [];
  String? error;
  @override
  void initState() { super.initState(); load(); }
  Future<void> load() async {
    try { final data = await widget.api.call('GET', '/v1/diaries'); if (mounted) setState(() { items = data as List; error = null; }); }
    catch (e) { if (mounted) setState(() => error = '$e'); }
  }
  Future<void> create() async {
    final title = TextEditingController();
    final accepted = await showDialog<bool>(context: context, builder: (ctx) => AlertDialog(
      title: const Text('Diary baru'), content: TextField(controller: title, decoration: const InputDecoration(labelText: 'Judul'), maxLength: 150),
      actions: [TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Batal')), FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Buat'))]));
    final value = title.text.trim(); title.dispose();
    if (accepted != true || value.isEmpty) return;
    try { await widget.api.call('POST', '/v1/diaries', {'title': value, 'notes': ''}); await load(); }
    catch (e) { if (mounted) setState(() => error = '$e'); }
  }
  @override
  Widget build(BuildContext context) => Scaffold(appBar: AppBar(title: const Text('Diary')),
    floatingActionButton: FloatingActionButton(onPressed: create, child: const Icon(Icons.add)),
    body: RefreshIndicator(onRefresh: load, child: ListView(padding: const EdgeInsets.all(16), children: [
      if (error != null) Text(error!),
      const Text('Kelompokkan scan dan catatan perkembangan. Hapus diary tidak menghapus Scan History.'),
      for (final item in items) Card(child: ListTile(title: Text('${item['title']}'), subtitle: Text('${item['created_at']}'),
        onTap: () async {
          await Navigator.push(context, MaterialPageRoute<void>(builder: (_) => DiaryDetail(api: widget.api, id: '${item['id']}', onScan: widget.onScan)));
          await load();
        })),
    ])));
}
class DiaryDetail extends StatefulWidget {
  const DiaryDetail({super.key, required this.api, required this.id, required this.onScan});
  final Api api; final String id; final void Function(String) onScan;
  @override
  State<DiaryDetail> createState() => _DiaryDetailState();
}
class _DiaryDetailState extends State<DiaryDetail> {
  final title = TextEditingController(); final notes = TextEditingController();
  List<dynamic> scans = []; String? message; bool busy = false;
  @override
  void initState() { super.initState(); load(); }
  @override
  void dispose() { title.dispose(); notes.dispose(); super.dispose(); }
  Future<void> action(Future<void> Function() fn) async {
    if (busy) return; setState(() { busy = true; message = null; });
    try { await fn(); } catch (e) { if (mounted) setState(() => message = '$e'); }
    finally { if (mounted) setState(() => busy = false); }
  }
  Future<void> load() async {
    try {
      final data = await widget.api.call('GET', '/v1/diaries/${widget.id}');
      if (mounted) setState(() { title.text = '${data['title']}'; notes.text = '${data['notes']}'; scans = data['scans'] as List; });
    } catch (e) { if (mounted) setState(() => message = '$e'); }
  }
  Future<void> save() async {
    await widget.api.call('PATCH', '/v1/diaries/${widget.id}', {'title': title.text.trim(), 'notes': notes.text});
  }
  Future<void> addScans() async {
    final history = await widget.api.call('GET', '/v1/scans') as List;
    final existing = scans.map((s) => '${s['id']}').toSet();
    final available = history.where((s) => !existing.contains('${s['id']}')).toList();
    if (!mounted) return;
    final selected = <String>{};
    final accepted = await showDialog<bool>(context: context, builder: (ctx) => StatefulBuilder(builder: (ctx, update) => AlertDialog(
      title: const Text('Pilih scan dari History'), content: SizedBox(width: 400, height: 350, child: ListView(children: [
        for (final s in available) CheckboxListTile(value: selected.contains('${s['id']}'), title: Text('${s['predicted_class'] ?? s['status']}'), subtitle: Text('${s['created_at']}'),
          onChanged: (v) => update(() { if (v == true) { selected.add('${s['id']}'); } else { selected.remove('${s['id']}'); } })),
      ])), actions: [TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Batal')), FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Tambahkan'))])));
    if (accepted == true) {
      for (final id in selected) { await widget.api.call('POST', '/v1/diaries/${widget.id}/scans', {'scan_id': id}); }
      await load();
    }
  }
  Future<void> pdf(bool share) async {
    await save();
    final bytes = await widget.api.pdf(widget.id);
    final directory = await getTemporaryDirectory();
    final file = File('${directory.path}/kulitku-${widget.id}.pdf'); await file.writeAsBytes(bytes);
    if (share) { await Share.shareXFiles([XFile(file.path)], text: 'Diary KulitKu'); }
    else { final opened = await OpenFilex.open(file.path); if (opened.type != ResultType.done) throw Exception(opened.message); }
  }
  Future<void> delete() async {
    final yes = await showDialog<bool>(context: context, builder: (ctx) => AlertDialog(title: const Text('Hapus diary?'),
      content: const Text('Scan aslinya tetap tersedia di History.'), actions: [TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Batal')),
      FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Hapus'))]));
    if (yes != true) return;
    await widget.api.call('DELETE', '/v1/diaries/${widget.id}'); if (mounted) Navigator.pop(context);
  }
  @override
  Widget build(BuildContext context) => Scaffold(appBar: AppBar(title: const Text('Detail diary')),
    body: ListView(padding: const EdgeInsets.all(20), children: [
      if (busy) const LinearProgressIndicator(), if (message != null) Text(message!),
      TextField(controller: title, maxLength: 150, decoration: const InputDecoration(labelText: 'Judul')),
      TextField(controller: notes, minLines: 3, maxLines: 8, maxLength: 20000, decoration: const InputDecoration(labelText: 'Catatan perkembangan')),
      FilledButton(onPressed: busy ? null : () => action(() async { await save(); if (mounted) setState(() => message = 'Diary tersimpan.'); }), child: const Text('Simpan')),
      Wrap(spacing: 8, children: [
        OutlinedButton(onPressed: busy ? null : () => action(() async { await save(); await addScans(); }), child: const Text('Pilih dari History')),
        OutlinedButton(onPressed: busy ? null : () => action(() async { await save(); widget.onScan(widget.id); }), child: const Text('Scan baru')),
      ]),
      for (final s in scans) Card(child: ListTile(title: Text('${s['predicted_class'] ?? s['status']}'), subtitle: Text('${s['created_at']}'),
        trailing: IconButton(tooltip: 'Keluarkan dari diary', onPressed: busy ? null : () => action(() async {
          await save(); await widget.api.call('DELETE', '/v1/diaries/${widget.id}/scans/${s['id']}'); await load();
        }), icon: const Icon(Icons.link_off)))),
      OutlinedButton(onPressed: busy ? null : () => action(() => pdf(false)), child: const Text('Export to PDF / Preview')),
      OutlinedButton(onPressed: busy ? null : () => action(() => pdf(true)), child: const Text('Export & Bagikan PDF')),
      TextButton(onPressed: busy ? null : () => action(delete), child: const Text('Hapus diary')),
    ]));
}
