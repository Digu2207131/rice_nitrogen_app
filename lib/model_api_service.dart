import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:mime/mime.dart';

class ModelApiService {
  static const String baseUrl = 'http://192.168.0.106:8000'; // PC LAN IP
  static const int timeoutSeconds = 60;

  /// Send image to Python backend and get SPAD prediction
  static Future<Map<String, dynamic>> predict(File imageFile) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/predict'),
      );

      // Detect MIME type automatically
      final mimeType = lookupMimeType(imageFile.path) ?? 'image/jpeg';
      final mimeSplit = mimeType.split('/');

      request.files.add(await http.MultipartFile.fromPath(
        'file',
        imageFile.path,
        contentType: MediaType(mimeSplit[0], mimeSplit[1]),
      ));

      print('Sending request to: $baseUrl/predict');
      print('With file: ${imageFile.path} (MIME: $mimeType)');

      var response =
          await request.send().timeout(Duration(seconds: timeoutSeconds));
      final responseBody = await response.stream.bytesToString();

      print("API Status: ${response.statusCode}");
      print("API Response: $responseBody");

      if (response.statusCode == 200) {
        return json.decode(responseBody);
      } else {
        return {'error': 'HTTP ${response.statusCode}: $responseBody'};
      }
    } catch (e) {
      print("API Request failed: $e");
      return {'error': 'Network error: $e'};
    }
  }
}
