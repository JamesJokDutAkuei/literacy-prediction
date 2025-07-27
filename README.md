# African Literacy Rate Prediction System

## Mission Statement
This project predicts literacy rates across 54 African countries using machine learning. Our Random Forest model analyzes gender, regional, and temporal factors to provide accurate literacy rate predictions for educational policy planning.

## Problem Description
Educational inequality remains a critical challenge across Africa. This system addresses the need for data-driven literacy rate predictions by analyzing World Bank statistics across African nations, enabling policymakers to make informed decisions about educational resource allocation and intervention strategies.

## 🌐 Public API Endpoint

**Base URL:** `https://literacy-prediction-production.up.railway.app`

**Swagger Documentation:** `https://literacy-prediction-production.up.railway.app/docs`

### Available Endpoints:
- `GET /` - API health check and information
- `GET /countries` - List all 54 African countries
- `POST /predict` - Single literacy rate prediction
- `POST /batch-predict` - Multiple predictions
- `GET /model-info` - ML model details and performance

### Example Prediction Request:
```json
POST /predict
{
  "country_code": "KEN",
  "gender": "Female", 
  "year": 2024
}
```

### Example Response:
```json
{
  "status": "success",
  "prediction": {
    "literacy_rate": 78.5,
    "country": {
      "code": "KEN",
      "name": "Kenya", 
      "region": "East Africa"
    },
    "gender": "Female",
    "year": 2024,
    "confidence": "Model R² = 0.762"
  },
  "model_used": "Random Forest"
}
```

## 📱 Mobile App Instructions

### Prerequisites:
- Flutter SDK (latest version)
- iOS Simulator or Android Emulator
- Xcode (for iOS) or Android Studio (for Android)

### Installation Steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/JamesJokDutAkuei/literacy-prediction.git
   cd literacy-prediction
   ```

2. **Navigate to Mobile App:**
   ```bash
   cd mobile_app
   ```

3. **Install Dependencies:**
   ```bash
   flutter pub get
   ```

4. **Run on iOS Simulator:**
   ```bash
   flutter run -d ios
   ```
   
5. **Run on Android Emulator:**
   ```bash
   flutter run -d android
   ```

6. **For Web (Optional):**
   ```bash
   flutter run -d chrome
   ```

### App Features:
- **Landing Screen:** Overview of literacy prediction system
- **Prediction Screen:** Select country, gender, and year for predictions
- **Results Screen:** Display predicted literacy rates with visual indicators
- **Real-time API Integration:** Connects to public FastAPI endpoint

### Troubleshooting:
- Ensure device/simulator is running before `flutter run`
- Check internet connection for API requests
- Run `flutter doctor` to verify Flutter installation

## 🧠 Machine Learning Model

**Algorithm:** Random Forest Regressor  
**Performance:** R² = 0.7623  
**Features:** Country, Gender, African Sub-region, Year, Interaction Terms  
**Training Data:** World Bank Gender Statistics (54 African Countries)

## 📊 Project Structure

```
literacy-prediction/
├── api/                          # FastAPI backend
│   ├── fastapi_app.py           # Main API application
│   ├── models/                  # Trained ML models
│   └── requirements_fastapi.txt # API dependencies
├── mobile_app/                  # Flutter mobile application
│   ├── lib/
│   │   ├── screens/            # App screens
│   │   ├── services/           # API integration
│   │   └── models/             # Data models
│   └── pubspec.yaml           # Flutter dependencies
├── notebooks/                  # Jupyter notebooks
│   └── african_literacy_analysis.ipynb
└── README.md                   # This file
```

## 🎥 Demo Video

**YouTube Link:** [5 Minute Demo](https://youtu.be/ZoQrpNn6pHs)

The demo video covers:
- API endpoint testing via Swagger UI
- Mobile app functionality demonstration
- Prediction accuracy showcase
- System architecture overview

## 🚀 Deployment

The API is deployed on Railway.app with automatic HTTPS and global CDN for optimal performance.

**Live API:** https://literacy-prediction-production.up.railway.app/docs

## 📈 Model Performance

- **Mean Squared Error:** 45.2
- **Root Mean Squared Error:** 6.7
- **Mean Absolute Error:** 4.9
- **R² Score:** 0.7623

## 🌍 Supported African Countries (54)

The system supports predictions for all 54 African Union member countries across 5 sub-regions:
- **North Africa** (6 countries)
- **West Africa** (16 countries)  
- **Central Africa** (9 countries)
- **East Africa** (18 countries)
- **Southern Africa** (5 countries)

## 📧 Contact

**Developer:** James Jok Dut Akuei  
**Repository:** https://github.com/JamesJokDutAkuei/literacy-prediction  
**API Documentation:** https://literacy-prediction-production.up.railway.app/docs
