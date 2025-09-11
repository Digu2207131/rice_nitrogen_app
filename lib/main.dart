import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'model_api_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Rice Nitrogen Analyzer',
      theme: ThemeData(
        primarySwatch: Colors.green,
        scaffoldBackgroundColor: Colors.grey[50],
        appBarTheme: AppBarTheme(
          backgroundColor: Colors.green[700],
          foregroundColor: Colors.white,
          elevation: 4,
        ),
      ),
      home: PredictionScreen(),
    );
  }
}

class PredictionScreen extends StatefulWidget {
  @override
  _PredictionScreenState createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  File? _pickedImage;
  bool _isLoading = false;
  double? _predictedSpad;
  String? _status;
  String? _suggestion;
  Color? _statusColor;

  Future<void> _pickImage(ImageSource source) async {
    try {
      final pickedFile = await ImagePicker().pickImage(source: source);
      if (pickedFile != null) {
        setState(() {
          _pickedImage = File(pickedFile.path);
          _predictedSpad = null;
          _status = null;
          _suggestion = null;
          _statusColor = null;
        });
        print("Image selected: ${pickedFile.path}");
      }
    } catch (e) {
      print("Error picking image: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error selecting image: $e')),
      );
    }
  }

  Future<void> _predict() async {
    if (_pickedImage == null) return;

    setState(() => _isLoading = true);

    try {
      print("Running server prediction...");

      Map<String, dynamic> response;

      // Call Python backend
      response = await ModelApiService.predict(_pickedImage!);

      if (response.containsKey('error')) {
        throw Exception(response['error']);
      }

      setState(() {
        _predictedSpad = response['prediction'];
        _status = response['status'];
        _suggestion = response['suggestion'];
        _updateStatusColor();
      });
    } catch (e) {
      print("Prediction failed: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Analysis failed: $e'),
          duration: Duration(seconds: 3),
        ),
      );
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _updateStatusColor() {
    if (_status == "Deficient") {
      _statusColor = Colors.red;
    } else if (_status == "Moderate") {
      _statusColor = Colors.orange;
    } else if (_status == "Sufficient") {
      _statusColor = Colors.green;
    } else {
      _statusColor = Colors.grey;
    }
  }

  double _calculateNitrogenPercentage() {
    if (_predictedSpad == null) return 0;
    return (_predictedSpad! / 60 * 100).clamp(0.0, 100.0);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.eco, color: Colors.white, size: 24),
            SizedBox(width: 8),
            Text('Rice Nitrogen Analyzer', style: TextStyle(fontSize: 18)),
          ],
        ),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          children: [
            // APP HEADER
            Container(
              padding: EdgeInsets.symmetric(vertical: 12),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.eco, size: 32, color: Colors.green[700]),
                      SizedBox(width: 10),
                      Text(
                        'RICE NITROGEN ANALYZER',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Colors.green[800],
                        ),
                      ),
                    ],
                  ),
                  SizedBox(height: 6),
                  Text(
                    'Smart tool for rice nitrogen management',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey[600],
                    ),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),

            SizedBox(height: 12),

            // IMAGE PREVIEW
            Container(
              height: 150,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey[200],
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.green[400]!, width: 2),
              ),
              child: _pickedImage != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.file(_pickedImage!, fit: BoxFit.cover),
                    )
                  : Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.image, size: 36, color: Colors.grey[400]),
                        SizedBox(height: 6),
                        Text('No image selected',
                            style: TextStyle(fontSize: 12, color: Colors.grey[500])),
                      ],
                    ),
            ),

            SizedBox(height: 16),

            // IMAGE SELECTION BUTTONS
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: Icon(Icons.camera_alt, size: 18),
                    label: Text('Take Photo', style: TextStyle(fontSize: 12)),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue[600],
                      foregroundColor: Colors.white,
                      minimumSize: Size(double.infinity, 44),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                ),
                SizedBox(width: 10),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: Icon(Icons.photo_library, size: 18),
                    label: Text('From Gallery', style: TextStyle(fontSize: 12)),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.green[600],
                      foregroundColor: Colors.white,
                      minimumSize: Size(double.infinity, 44),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                ),
              ],
            ),

            SizedBox(height: 16),

            // ANALYSIS BUTTON
            Container(
              width: double.infinity,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [Colors.green[600]!, Colors.blue[600]!],
                  begin: Alignment.centerLeft,
                  end: Alignment.centerRight,
                ),
                borderRadius: BorderRadius.circular(10),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black26,
                    blurRadius: 4,
                    offset: Offset(0, 2),
                  )
                ],
              ),
              child: ElevatedButton(
                onPressed: _isLoading ? null : (_pickedImage != null ? _predict : null),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.transparent,
                  foregroundColor: Colors.white,
                  shadowColor: Colors.transparent,
                  minimumSize: Size(double.infinity, 48),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.analytics, size: 20),
                    SizedBox(width: 6),
                    Text('Analyze Nitrogen', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
                  ],
                ),
              ),
            ),

            SizedBox(height: 16),

            // LOADING INDICATOR
            if (_isLoading) ...[
              Column(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 8),
                  Text(
                    'Analyzing image... This may take 20-30 seconds',
                    style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ],

            // ANALYSIS RESULTS - COMPACT
            if (_predictedSpad != null) ...[
              SizedBox(height: 16),
              Container(
                padding: EdgeInsets.all(10), // smaller padding
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(10),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black12,
                      blurRadius: 4,
                      offset: Offset(0, 2),
                    )
                  ],
                ),
                child: Column(
                  children: [
                    Text(
                      'ANALYSIS RESULTS',
                      style: TextStyle(
                        fontSize: 13, // smaller
                        fontWeight: FontWeight.bold,
                        color: Colors.green[800],
                      ),
                    ),
                    SizedBox(height: 8),

                    // Nitrogen Status
                    Container(
                      padding: EdgeInsets.symmetric(vertical: 8, horizontal: 10),
                      decoration: BoxDecoration(
                        color: (_statusColor ?? Colors.grey).withOpacity(0.1),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: _statusColor ?? Colors.grey, width: 1.5),
                      ),
                      child: Column(
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                _status == "Deficient" ? Icons.warning : 
                                _status == "Moderate" ? Icons.info_outline : 
                                Icons.check_circle,
                                color: _statusColor,
                                size: 18,
                              ),
                              SizedBox(width: 4),
                              Text(
                                'NITROGEN STATUS',
                                style: TextStyle(
                                  fontSize: 11, // smaller
                                  fontWeight: FontWeight.bold,
                                  color: Colors.grey[700],
                                ),
                              ),
                            ],
                          ),
                          SizedBox(height: 4),
                          Text(
                            _status ?? '',
                            style: TextStyle(
                              fontSize: 16, // smaller SPAD display
                              fontWeight: FontWeight.bold,
                              color: _statusColor ?? Colors.grey,
                            ),
                          ),
                          SizedBox(height: 4),
                          // Nitrogen Percentage with Progress Bar
                          Column(
                            children: [
                              Text(
                                '${_calculateNitrogenPercentage().toStringAsFixed(1)}% Nitrogen',
                                style: TextStyle(
                                  fontSize: 14,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.grey[800],
                                ),
                              ),
                              SizedBox(height: 4),
                              LinearProgressIndicator(
                                value: _calculateNitrogenPercentage() / 100,
                                backgroundColor: Colors.grey[300],
                                valueColor: AlwaysStoppedAnimation<Color>(_statusColor ?? Colors.grey),
                                minHeight: 5,
                                borderRadius: BorderRadius.circular(2.5),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    SizedBox(height: 8),

                    // SPAD Value - Compact
                    Container(
                      padding: EdgeInsets.symmetric(vertical: 6, horizontal: 10),
                      decoration: BoxDecoration(
                        color: Colors.grey[100],
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: Colors.green[400]!, width: 1),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            'SPAD Value:',
                            style: TextStyle(
                              fontSize: 12, // smaller
                              fontWeight: FontWeight.bold,
                              color: Colors.grey[700],
                            ),
                          ),
                          Text(
                            _predictedSpad!.toStringAsFixed(2),
                            style: TextStyle(
                              fontSize: 18, // smaller
                              fontWeight: FontWeight.bold,
                              color: Colors.green[800],
                            ),
                          ),
                        ],
                      ),
                    ),
                    SizedBox(height: 8),

                    // Recommendation Section - Compact
                    Container(
                      width: double.infinity,
                      padding: EdgeInsets.all(8), // reduced padding
                      decoration: BoxDecoration(
                        color: Colors.lightGreen[50],
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: Colors.lightGreen[400]!, width: 1.5),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Icon(Icons.local_florist, size: 16, color: Colors.green[700]),
                              SizedBox(width: 4),
                              Text(
                                'RECOMMENDATION',
                                style: TextStyle(
                                  fontSize: 12,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.green[800],
                                ),
                              ),
                            ],
                          ),
                          SizedBox(height: 4),
                          Text(
                            _suggestion ?? '',
                            style: TextStyle(
                              fontSize: 11, // smaller
                              fontWeight: FontWeight.w500,
                              color: Colors.green[900],
                              height: 1.2,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ],

            SizedBox(height: 20),

            // FOOTER
            Container(
              padding: EdgeInsets.symmetric(vertical: 12, horizontal: 10),
              decoration: BoxDecoration(
                color: Colors.green[50],
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.green[400]!, width: 2),
              ),
              child: Column(
                children: [
                  Text(
                    'Developed by',
                    style: TextStyle(
                      fontSize: 11,
                      color: Colors.grey[600],
                    ),
                  ),
                  SizedBox(height: 4),
                  Text(
                    'Dept. of Agricultural & Industrial Engineering, HSTU',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                      color: Colors.green[900],
                    ),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
