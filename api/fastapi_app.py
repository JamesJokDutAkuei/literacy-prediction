# African Literacy Rate Prediction API
# FastAPI REST API for predicting literacy rates in African countries

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="African Literacy Rate Prediction API",
    description="Machine Learning API for predicting literacy rates across 54 African countries using Random Forest model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    country_code: str = Field(..., description="ISO 3-letter country code (e.g., KEN, NGA, ZAF)", example="KEN")
    gender: str = Field(..., description="Gender category", example="Female", pattern="^(Male|Female)$")
    year: int = Field(..., description="Target year for prediction", example=2024, ge=1960, le=2030)

class CountryInfo(BaseModel):
    code: str
    name: str
    region: str

class PredictionResponse(BaseModel):
    status: str
    prediction: dict
    model_used: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest]

class ModelInfoResponse(BaseModel):
    model_type: str
    african_countries: int
    african_subregions: List[str]
    supported_years: str
    supported_genders: List[str]
    performance: dict
    features: List[str]
    training_date: str

class CountriesResponse(BaseModel):
    countries: List[CountryInfo]
    total_countries: int

# Load trained model and preprocessing components
try:
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    model = joblib.load(os.path.join(script_dir, 'models/literacy_model.pkl'))
    scaler = joblib.load(os.path.join(script_dir, 'models/scaler.pkl'))
    label_encoders = joblib.load(os.path.join(script_dir, 'models/label_encoders.pkl'))
    metadata = joblib.load(os.path.join(script_dir, 'models/model_metadata.pkl'))
    print("✅ Model and preprocessing components loaded successfully!")
    print(f"🏆 Loaded model: {metadata.get('model_type', 'Unknown')} (R² = {metadata['model_performance']['R²']:.4f})")
except Exception as e:
    print(f"❌ Error loading model components: {e}")
    raise

# African countries mapping
african_countries_map = {
    'DZA': {'name': 'Algeria', 'region': 'North Africa'},
    'AGO': {'name': 'Angola', 'region': 'Central Africa'},
    'BEN': {'name': 'Benin', 'region': 'West Africa'},
    'BWA': {'name': 'Botswana', 'region': 'Southern Africa'},
    'BFA': {'name': 'Burkina Faso', 'region': 'West Africa'},
    'BDI': {'name': 'Burundi', 'region': 'East Africa'},
    'CMR': {'name': 'Cameroon', 'region': 'Central Africa'},
    'CPV': {'name': 'Cabo Verde', 'region': 'West Africa'},
    'CAF': {'name': 'Central African Republic', 'region': 'Central Africa'},
    'TCD': {'name': 'Chad', 'region': 'Central Africa'},
    'COM': {'name': 'Comoros', 'region': 'East Africa'},
    'COG': {'name': 'Congo, Rep.', 'region': 'Central Africa'},
    'COD': {'name': 'Congo, Dem. Rep.', 'region': 'Central Africa'},
    'CIV': {'name': "Cote d'Ivoire", 'region': 'West Africa'},
    'DJI': {'name': 'Djibouti', 'region': 'East Africa'},
    'EGY': {'name': 'Egypt, Arab Rep.', 'region': 'North Africa'},
    'GNQ': {'name': 'Equatorial Guinea', 'region': 'Central Africa'},
    'ERI': {'name': 'Eritrea', 'region': 'East Africa'},
    'ETH': {'name': 'Ethiopia', 'region': 'East Africa'},
    'GAB': {'name': 'Gabon', 'region': 'Central Africa'},
    'GMB': {'name': 'Gambia, The', 'region': 'West Africa'},
    'GHA': {'name': 'Ghana', 'region': 'West Africa'},
    'GIN': {'name': 'Guinea', 'region': 'West Africa'},
    'GNB': {'name': 'Guinea-Bissau', 'region': 'West Africa'},
    'KEN': {'name': 'Kenya', 'region': 'East Africa'},
    'LSO': {'name': 'Lesotho', 'region': 'Southern Africa'},
    'LBR': {'name': 'Liberia', 'region': 'West Africa'},
    'LBY': {'name': 'Libya', 'region': 'North Africa'},
    'MDG': {'name': 'Madagascar', 'region': 'East Africa'},
    'MWI': {'name': 'Malawi', 'region': 'East Africa'},
    'MLI': {'name': 'Mali', 'region': 'West Africa'},
    'MRT': {'name': 'Mauritania', 'region': 'West Africa'},
    'MUS': {'name': 'Mauritius', 'region': 'East Africa'},
    'MAR': {'name': 'Morocco', 'region': 'North Africa'},
    'MOZ': {'name': 'Mozambique', 'region': 'East Africa'},
    'NAM': {'name': 'Namibia', 'region': 'Southern Africa'},
    'NER': {'name': 'Niger', 'region': 'West Africa'},
    'NGA': {'name': 'Nigeria', 'region': 'West Africa'},
    'RWA': {'name': 'Rwanda', 'region': 'East Africa'},
    'STP': {'name': 'Sao Tome and Principe', 'region': 'Central Africa'},
    'SEN': {'name': 'Senegal', 'region': 'West Africa'},
    'SYC': {'name': 'Seychelles', 'region': 'East Africa'},
    'SLE': {'name': 'Sierra Leone', 'region': 'West Africa'},
    'SOM': {'name': 'Somalia', 'region': 'East Africa'},
    'ZAF': {'name': 'South Africa', 'region': 'Southern Africa'},
    'SSD': {'name': 'South Sudan', 'region': 'East Africa'},
    'SDN': {'name': 'Sudan', 'region': 'North Africa'},
    'SWZ': {'name': 'Eswatini', 'region': 'Southern Africa'},
    'TZA': {'name': 'Tanzania', 'region': 'East Africa'},
    'TGO': {'name': 'Togo', 'region': 'West Africa'},
    'TUN': {'name': 'Tunisia', 'region': 'North Africa'},
    'UGA': {'name': 'Uganda', 'region': 'East Africa'},
    'ZMB': {'name': 'Zambia', 'region': 'East Africa'},
    'ZWE': {'name': 'Zimbabwe', 'region': 'East Africa'}
}

# Helper function to get subregion mapping
subregion_mapping = {
    'Algeria': 'North Africa', 'Egypt, Arab Rep.': 'North Africa', 'Libya': 'North Africa',
    'Morocco': 'North Africa', 'Sudan': 'North Africa', 'Tunisia': 'North Africa',
    'Benin': 'West Africa', 'Burkina Faso': 'West Africa', 'Cabo Verde': 'West Africa',
    "Cote d'Ivoire": 'West Africa', 'Gambia, The': 'West Africa', 'Ghana': 'West Africa',
    'Guinea': 'West Africa', 'Guinea-Bissau': 'West Africa', 'Liberia': 'West Africa',
    'Mali': 'West Africa', 'Mauritania': 'West Africa', 'Niger': 'West Africa',
    'Nigeria': 'West Africa', 'Senegal': 'West Africa', 'Sierra Leone': 'West Africa',
    'Togo': 'West Africa',
    'Angola': 'Central Africa', 'Cameroon': 'Central Africa', 'Central African Republic': 'Central Africa',
    'Chad': 'Central Africa', 'Congo, Rep.': 'Central Africa', 'Congo, Dem. Rep.': 'Central Africa',
    'Equatorial Guinea': 'Central Africa', 'Gabon': 'Central Africa', 'Sao Tome and Principe': 'Central Africa',
    'Burundi': 'East Africa', 'Comoros': 'East Africa', 'Djibouti': 'East Africa',
    'Eritrea': 'East Africa', 'Ethiopia': 'East Africa', 'Kenya': 'East Africa',
    'Madagascar': 'East Africa', 'Malawi': 'East Africa', 'Mauritius': 'East Africa',
    'Mozambique': 'East Africa', 'Rwanda': 'East Africa', 'Seychelles': 'East Africa',
    'Somalia': 'East Africa', 'South Sudan': 'East Africa', 'Tanzania': 'East Africa',
    'Uganda': 'East Africa', 'Zambia': 'East Africa', 'Zimbabwe': 'East Africa',
    'Botswana': 'Southern Africa', 'Eswatini': 'Southern Africa', 'Lesotho': 'Southern Africa',
    'Namibia': 'Southern Africa', 'South Africa': 'Southern Africa'
}

@app.get("/", summary="Health Check", description="Root endpoint for API health check")
async def root():
    """Root endpoint for API health check."""
    return {
        "message": "🌍 African Literacy Rate Prediction API",
        "status": "✅ API is running successfully!",
        "version": "1.0.0",
        "model_info": {
            "type": metadata.get('model_type', 'Unknown'),
            "accuracy": f"R² = {metadata['model_performance']['R²']:.4f}",
            "countries": len(african_countries_map),
            "last_updated": metadata.get('training_date', 'Unknown')
        },
        "endpoints": {
            "predict": "/predict",
            "countries": "/countries",
            "model_info": "/model-info",
            "batch_predict": "/batch-predict",
            "docs": "/docs"
        }
    }

@app.get("/countries", response_model=CountriesResponse, summary="Get African Countries", 
         description="Retrieve list of all 54 African countries with regional classifications")
async def get_countries():
    """Get list of all African countries with regional information."""
    countries = [
        CountryInfo(code=code, name=info['name'], region=info['region'])
        for code, info in african_countries_map.items()
    ]
    
    return CountriesResponse(
        countries=countries,
        total_countries=len(countries)
    )

@app.get("/model-info", response_model=ModelInfoResponse, summary="Get Model Information",
         description="Retrieve detailed information about the ML model and its performance")
async def get_model_info():
    """Get detailed model information and performance metrics."""
    return ModelInfoResponse(
        model_type=metadata.get('model_type', 'Random Forest'),
        african_countries=len(african_countries_map),
        african_subregions=['North Africa', 'West Africa', 'Central Africa', 'East Africa', 'Southern Africa'],
        supported_years="1960-2030",
        supported_genders=["Male", "Female"],
        performance=metadata.get('model_performance', {}),
        features=metadata.get('features', []),
        training_date=metadata.get('training_date', datetime.now().strftime('%Y-%m-%d'))
    )

@app.post("/predict", response_model=PredictionResponse, summary="Predict Literacy Rate",
          description="Predict literacy rate for a specific country, gender, and year")
async def predict_literacy(request: PredictionRequest):
    """
    Predict literacy rate for given parameters.
    
    - **country_code**: ISO 3-letter country code (e.g., KEN, NGA, ZAF)
    - **gender**: Gender category (Male or Female)
    - **year**: Target year for prediction (1960-2030)
    """
    try:
        country_code = request.country_code.upper()
        gender = request.gender.title()
        year = request.year
        
        # Validate country code
        if country_code not in african_countries_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid country code: {country_code}. Must be one of the 54 African countries."
            )
        
        country_info = african_countries_map[country_code]
        country_name = country_info['name']
        subregion = country_info['region']
        
        # Prepare input data for prediction using correct column names
        input_data = pd.DataFrame({
            'REF_AREA': [country_code],
            'SEX_LABEL': [gender],
            'African_Subregion': [subregion],
            'TIME_PERIOD': [year]
        })
        
        # Create additional features (matching training pipeline)
        input_data['Year_Normalized'] = (input_data['TIME_PERIOD'] - 1990) / 30.0
        input_data['Gender_Year_Interaction'] = input_data['Year_Normalized'] * (input_data['SEX_LABEL'] == 'Female').astype(int)
        
        # Create subregion-year interaction
        subregion_encoded = label_encoders['African_Subregion'].transform([subregion])[0]
        input_data['Subregion_Year_Interaction'] = input_data['Year_Normalized'] * subregion_encoded
        
        # Apply label encoding and map to expected feature names
        input_data['Country_Encoded'] = label_encoders['REF_AREA'].transform(input_data['REF_AREA'])
        input_data['Gender_Encoded'] = label_encoders['SEX_LABEL'].transform(input_data['SEX_LABEL'])
        input_data['Subregion_Encoded'] = label_encoders['African_Subregion'].transform(input_data['African_Subregion'])
        
        # Prepare features for prediction using the expected column names
        feature_columns = ['Country_Encoded', 'Gender_Encoded', 'Subregion_Encoded', 
                          'TIME_PERIOD', 'Year_Normalized', 'Gender_Year_Interaction', 
                          'Subregion_Year_Interaction']
        
        X = input_data[feature_columns]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        prediction = max(0, min(100, prediction))  # Ensure prediction is between 0-100
        
        response_data = {
            "status": "success",
            "prediction": {
                "literacy_rate": round(prediction, 1),
                "country": {
                    "code": country_code,
                    "name": country_name,
                    "region": subregion
                },
                "gender": gender,
                "year": year,
                "confidence": f"Model R² = {metadata['model_performance']['R²']:.3f}"
            },
            "model_used": metadata.get('model_type', 'Random Forest')
        }
        
        return PredictionResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict", summary="Batch Predictions",
          description="Process multiple prediction requests at once")
async def batch_predict(request: BatchPredictionRequest):
    """Process multiple prediction requests in a single API call."""
    try:
        results = []
        
        for pred_request in request.predictions:
            try:
                # Use the individual predict function
                result = await predict_literacy(pred_request)
                results.append(result.dict())
            except HTTPException as e:
                results.append({
                    "status": "error",
                    "error": e.detail,
                    "request": pred_request.dict()
                })
        
        return {
            "status": "completed",
            "total_requests": len(request.predictions),
            "successful_predictions": len([r for r in results if r.get("status") == "success"]),
            "failed_predictions": len([r for r in results if r.get("status") == "error"]),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (for deployment) or default to 5000
    port = int(os.getenv("PORT", 5000))
    
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
