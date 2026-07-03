from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Kano Maternal Triage API")

# Allow the React frontend to communicate with this Python backend from ANY internet location
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

class PatientData(BaseModel):
    age: int
    previous_pregnancies: int
    anc_visits: int
    pregnancy_complications: str
    education_level: str
    settlement_type: str
    place_delivered: str

MODEL_PATH = '02_optimized_lightgbm_pipeline.pkl'
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        MODEL_LOADED = True
        print("✅ LightGBM Model successfully loaded!")
    else:
        model = None
        MODEL_LOADED = False
        print("⚠️ Model file not found. API cannot serve predictions.")
except Exception as e:
    model = None
    MODEL_LOADED = False
    print(f"⚠️ Error loading model: {e}")

# 🚨 PRACTICAL FIX: Cloud servers ping the root URL to see if the app is alive. 
@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": MODEL_LOADED, "message": "ACEPHAP Triage Engine is running."}

@app.post("/predict")
async def predict_risk(patient: PatientData):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="AI Engine is offline. Model file not found.")

    try:
        # 1. Convert incoming JSON to a Pandas DataFrame
        input_data = pd.DataFrame([{
            'age': patient.age,
            'previous_pregnancies': patient.previous_pregnancies,
            'anc_visits': patient.anc_visits,
            'pregnancy_complications': patient.pregnancy_complications,
            'education_level': patient.education_level,
            'settlement_type': patient.settlement_type,
            'place_delivered': patient.place_delivered,
            'employment_status': 'formally_employed' # Defaulted to match 8-feature ML matrix
        }])

        drivers = []

        # 2. Run Actual ML Inference
        probabilities = model.predict_proba(input_data)
        risk_score = probabilities[0][1] 
        
        # Simulated SHAP driver mapping based on Phase 6 insights
        if patient.pregnancy_complications == 'yes':
            drivers.append({"factor": "Existing Complications", "impact": "Critical Risk Increase (SHAP: +1.44)"})
        if patient.previous_pregnancies > 4:
            drivers.append({"factor": f"High Parity ({patient.previous_pregnancies})", "impact": "High Risk Increase (SHAP: +3.83)"})
        if patient.anc_visits == 0:
            drivers.append({"factor": "Zero ANC Visits", "impact": "High Risk Increase (SHAP: +2.62)"})
        if patient.place_delivered == 'enroute':
            drivers.append({"factor": "Enroute Delivery", "impact": "Severe Acute Risk"})

        # 3. Format the exact JSON response expected by React
        return {
            "probability": f"{risk_score * 100:.1f}",
            "classification": "High Mortality Risk" if risk_score >= 0.50 else "Standard Risk",
            "drivers": drivers if len(drivers) > 0 else [{"factor": "Standard Profile", "impact": "No acute drivers detected"}]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))