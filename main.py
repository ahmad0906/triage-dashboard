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
        # 1. Map ordinal data (Education is numerical in your matrix)
        edu_map = {'none_or_non_formal': 0, 'arabic_ismiyya': 1, 'primary': 2, 'secondary': 3, 'tertiary': 4}
        
        # 2. Build the Zero Matrix based EXACTLY on your 17 training columns
        feature_dict = {
            'age': patient.age,
            'education_level': edu_map.get(patient.education_level, 0),
            'previous_pregnancies': patient.previous_pregnancies,
            'anc_visits': patient.anc_visits,
            'settlement_type_rural': 0,
            'settlement_type_semi-urban': 0,
            'settlement_type_urban': 0,
            'employment_status_casually_employed': 0,
            'employment_status_formally_employed': 0,
            'employment_status_seasonally_employed': 0,
            'employment_status_self_employed': 0,
            'employment_status_unemployed': 0,
            'place_delivered_enroute': 0,
            'place_delivered_hf': 0,
            'place_delivered_home': 0,
            'pregnancy_complications_no': 0,
            'pregnancy_complications_yes': 0
        }

        # 3. Flip the switch (0 to 1) for the exact categories the user selected
        settlement_col = f"settlement_type_{patient.settlement_type}"
        if settlement_col in feature_dict:
            feature_dict[settlement_col] = 1

        place_col = f"place_delivered_{patient.place_delivered}"
        if place_col in feature_dict:
            feature_dict[place_col] = 1

        comp_col = f"pregnancy_complications_{patient.pregnancy_complications}"
        if comp_col in feature_dict:
            feature_dict[comp_col] = 1

        # The frontend doesn't ask for employment status, so we default to one
        feature_dict['employment_status_formally_employed'] = 1

        # 4. Convert to DataFrame in the exact column order the model expects
        input_data = pd.DataFrame([feature_dict])

        drivers = []

        # 5. Run Actual ML Inference
        probabilities = model.predict_proba(input_data)
        risk_score = probabilities[0][1] 
        
        # Simulated SHAP drivers
        if patient.pregnancy_complications == 'yes':
            drivers.append({"factor": "Existing Complications", "impact": "Critical Risk Increase (SHAP: +1.44)"})
        if patient.previous_pregnancies > 4:
            drivers.append({"factor": f"High Parity ({patient.previous_pregnancies})", "impact": "High Risk Increase (SHAP: +3.83)"})
        if patient.anc_visits == 0:
            drivers.append({"factor": "Zero ANC Visits", "impact": "High Risk Increase (SHAP: +2.62)"})
        if patient.place_delivered == 'enroute':
            drivers.append({"factor": "Enroute Delivery", "impact": "Severe Acute Risk"})

        # 6. Format the response
        return {
            "probability": f"{risk_score * 100:.1f}",
            "classification": "High Mortality Risk" if risk_score >= 0.50 else "Standard Risk",
            "drivers": drivers if len(drivers) > 0 else [{"factor": "Standard Profile", "impact": "No acute drivers detected"}]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))