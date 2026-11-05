import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

class Api {
  Api(this.baseUrl, this.headers);
  final String baseUrl;
  final Future<Map<String, String>> Function() headers;
  Future<dynamic> call(String method, String path, [Map<String, dynamic>? body]) async {
    final request = http.Request(method, Uri.parse('$baseUrl$path'));
    request.headers.addAll(await headers());
    if (body != null) { request.headers['Content-Type'] = 'application/json'; request.body = jsonEncode(body); }
    final streamed = await request.send().timeout(const Duration(seconds: 60));
    final response = await http.Response.fromStream(streamed).timeout(const Duration(seconds: 60));
    if (response.statusCode >= 400) { throw Exception('API ${response.statusCode}: ${response.body}'); }
    return jsonDecode(response.body);
  }
  Future<Uint8List> pdf(String diaryId) async {
    final response = await http.get(Uri.parse('$baseUrl/v1/diaries/$diaryId/pdf'), headers: await headers()).timeout(const Duration(seconds: 120));
    if (response.statusCode != 200) { throw Exception('PDF gagal: ${response.statusCode}'); }
    return response.bodyBytes;
  }
}
