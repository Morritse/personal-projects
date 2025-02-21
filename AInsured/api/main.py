from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="AInsued API", version="0.1.0")

class HealthInfo(BaseModel):
    age: int
    state: str
    smoker: bool
    bmi: float
    exercise_frequency: Optional[int] = None  # days per week
    chronic_conditions: Optional[List[str]] = []

class InsurancePlan(BaseModel):
    plan_id: str
    name: str
    premium: float
    deductible: float
    out_of_pocket_max: float
    coverage_level: str

class CostPrediction(BaseModel):
    expected_annual_cost: float
    recommended_plans: List[InsurancePlan]
    confidence_score: float

@app.get("/")
async def root():
    return {"message": "Welcome to AInsued API"}

@app.post("/predict", response_model=CostPrediction)
async def predict_costs(health_info: HealthInfo):
    """
    Predict healthcare costs and recommend insurance plans based on user health information
    """
    try:
        # TODO: Implement actual prediction logic
        # This is a placeholder response
        return CostPrediction(
            expected_annual_cost=5000.0,
            recommended_plans=[
                InsurancePlan(
                    plan_id="basic_plan",
                    name="Basic Coverage",
                    premium=300.0,
                    deductible=2000.0,
                    out_of_pocket_max=8000.0,
                    coverage_level="Silver"
                )
            ],
            confidence_score=0.85
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
