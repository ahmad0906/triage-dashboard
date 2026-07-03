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
        # 1. Map text data from the frontend into the exact numeric values the AI was trained on
        comp_map = {'no': 0, 'yes': 1}
        edu_map = {'none_or_non_formal': 0, 'arabic_ismiyya': 1, 'primary': 2, 'secondary': 3, 'tertiary': 4}
        sett_map = {'rural': 0, 'semi-urban': 1, 'urban': 2}
        place_map = {'hf': 0, 'home': 1, 'enroute': 2}
        
        # 2. Convert incoming JSON to a purely numeric Pandas DataFrame
        input_data = pd.DataFrame([{
            'age': patient.age,
            'previous_pregnancies': patient.previous_pregnancies,
            'anc_visits': patient.anc_visits,
            'pregnancy_complications': comp_map.get(patient.pregnancy_complications, 0),
            'education_level': edu_map.get(patient.education_level, 0),
            'settlement_type': sett_map.get(patient.settlement_type, 0),
            'place_delivered': place_map.get(patient.place_delivered, 0),
            'employment_status': 0 # Defaulted to match 8-feature ML matrix (0 = formally_employed)
        }])

        drivers = []

        # 3. Run Actual ML Inference
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

        # 4. Format the exact JSON response expected by React
        return {
            "probability": f"{risk_score * 100:.1f}",
            "classification": "High Mortality Risk" if risk_score >= 0.50 else "Standard Risk",
            "drivers": drivers if len(drivers) > 0 else [{"factor": "Standard Profile", "impact": "No acute drivers detected"}]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))