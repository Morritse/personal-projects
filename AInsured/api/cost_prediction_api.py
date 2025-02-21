from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.cost_predictor import HealthcareCostPredictor

app = FastAPI(title="AInsued Cost Prediction API", version="0.1.0")

# Initialize and train the predictor
predictor = HealthcareCostPredictor()
predictor.train_models()

class PhysicianPredictionRequest(BaseModel):
    provider_type: str = Field(..., description="Type of healthcare provider")
    procedure_code: str = Field(..., description="HCPCS procedure code")
    place_of_service: str = Field(..., description="Place where service was provided")
    num_beneficiaries: int = Field(..., description="Number of beneficiaries", gt=0)
    num_services: int = Field(..., description="Number of services provided", gt=0)
    submitted_charge: float = Field(..., description="Submitted charge amount", gt=0)
    allowed_amount: float = Field(..., description="Medicare allowed amount", gt=0)

class HospitalPredictionRequest(BaseModel):
    drg_code: str = Field(..., description="Diagnosis Related Group (DRG) code")
    num_discharges: int = Field(..., description="Number of discharges", gt=0)
    submitted_charge: float = Field(..., description="Submitted charge amount", gt=0)

class PredictionResponse(BaseModel):
    predicted_cost: float
    confidence_score: float
    factors: List[dict]

@app.get("/")
async def root():
    return {
        "message": "Welcome to AInsued Cost Prediction API",
        "version": "0.1.0",
        "endpoints": [
            "/predict/physician",
            "/predict/hospital",
            "/categories"
        ]
    }

@app.get("/categories")
async def get_categories():
    """Get available categories for predictions"""
    return {
        "physician": {
            "provider_types": predictor.physician_categories['Rndrng_Prvdr_Type'],
            "procedure_codes": predictor.physician_categories['HCPCS_Cd'],
            "places_of_service": predictor.physician_categories['Place_Of_Srvc']
        },
        "hospital": {
            "drg_codes": predictor.hospital_categories['DRG_Cd']
        }
    }

@app.post("/predict/physician", response_model=PredictionResponse)
async def predict_physician_cost(request: PhysicianPredictionRequest):
    """Predict physician service cost"""
    try:
        # Validate categories
        if request.provider_type not in predictor.physician_categories['Rndrng_Prvdr_Type']:
            raise HTTPException(status_code=400, detail=f"Invalid provider type. Use /categories to see valid options.")
        if request.procedure_code not in predictor.physician_categories['HCPCS_Cd']:
            raise HTTPException(status_code=400, detail=f"Invalid procedure code. Use /categories to see valid options.")
        if request.place_of_service not in predictor.physician_categories['Place_Of_Srvc']:
            raise HTTPException(status_code=400, detail=f"Invalid place of service. Use /categories to see valid options.")
            
        # Make prediction
        predicted_cost = predictor.predict_physician_cost(
            provider_type=request.provider_type,
            procedure_code=request.procedure_code,
            place_of_service=request.place_of_service,
            num_beneficiaries=request.num_beneficiaries,
            num_services=request.num_services,
            submitted_charge=request.submitted_charge,
            allowed_amount=request.allowed_amount
        )
        
        # Get feature importance for this prediction
        factors = [
            {"name": "Allowed Amount", "importance": 0.889698},
            {"name": "Submitted Charge", "importance": 0.091326},
            {"name": "Procedure Code", "importance": 0.008490},
            {"name": "Number of Services", "importance": 0.007496},
            {"name": "Number of Beneficiaries", "importance": 0.002649},
            {"name": "Provider Type", "importance": 0.000258},
            {"name": "Place of Service", "importance": 0.000083}
        ]
        
        return {
            "predicted_cost": round(predicted_cost, 2),
            "confidence_score": 0.9982,  # Using test R² score
            "factors": factors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/hospital", response_model=PredictionResponse)
async def predict_hospital_cost(request: HospitalPredictionRequest):
    """Predict hospital service cost"""
    try:
        # Validate categories
        if request.drg_code not in predictor.hospital_categories['DRG_Cd']:
            raise HTTPException(status_code=400, detail=f"Invalid DRG code. Use /categories to see valid options.")
            
        # Make prediction
        predicted_cost = predictor.predict_hospital_cost(
            drg_code=request.drg_code,
            num_discharges=request.num_discharges,
            submitted_charge=request.submitted_charge
        )
        
        # Get feature importance for this prediction
        factors = [
            {"name": "Submitted Charge", "importance": 0.646740},
            {"name": "DRG Code", "importance": 0.303087},
            {"name": "Number of Discharges", "importance": 0.050173}
        ]
        
        return {
            "predicted_cost": round(predicted_cost, 2),
            "confidence_score": 0.8263,  # Using test R² score
            "factors": factors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)  # Changed host to 0.0.0.0 to allow external access
