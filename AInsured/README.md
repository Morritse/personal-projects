# AInsued - AI-Powered Insurance Recommendation System

## Overview
AInsued analyzes Medicare data to predict healthcare costs and recommend optimal insurance plans based on individual factors. The system uses real CMS (Centers for Medicare & Medicaid Services) data to provide personalized insurance recommendations.

## Data Sources
The system analyzes three key Medicare datasets:

1. Geographic Variation Data
- Regional cost variations
- Utilization patterns
- Risk scores
- Average costs: $9,338 per beneficiary annually

2. Claims Pattern Data
- Common claim types
- Error rates
- Provider patterns
- Service utilization

3. Enrollment Data
- Plan selection patterns
- Demographics
- Coverage types
- ~54M total beneficiaries

## Features

### Cost Prediction
Predicts annual healthcare costs based on:
- Age
- Geographic location
- Chronic conditions
- Income level
- Regional cost variations

### Plan Recommendations
Recommends insurance plans considering:
- Predicted costs
- Health status
- Age
- Regional factors
- Provider networks

### Available Plans
1. Original Medicare
- Traditional fee-for-service
- Flexibility in provider choice
- Higher out-of-pocket costs

2. Medicare Advantage
- All-in-one coverage
- Network restrictions
- Additional benefits
- ~28% of beneficiaries

3. Medigap + Part D
- Comprehensive coverage
- Best for chronic conditions
- Higher premiums
- Separate drug coverage

## Usage

```python
from data.data_analyzer import InsuranceCostAnalyzer

# Initialize analyzer
analyzer = InsuranceCostAnalyzer()

# Get cost prediction
prediction = analyzer.predict_costs(
    age=70,
    state="CA",
    has_chronic_conditions=True,
    income_level="medium"
)

# Get plan recommendations
recommendations = analyzer.recommend_plans(
    age=70,
    state="CA",
    predicted_costs=prediction['predicted_annual_cost'],
    has_chronic_conditions=True
)
```

## Key Findings

### Cost Patterns
- Average annual Medicare spending: $9,338
- Inpatient costs: $3,675 average
- Outpatient costs: $1,851 average
- Emergency costs: $511 per 1000 beneficiaries

### Enrollment Insights
- Total beneficiaries: 54 million
- Medicare Advantage: 15.2 million (28%)
- Prescription coverage: 36.7 million (68%)
- Dual eligible: 9.9 million (18%)

### Risk Factors
- Age significantly impacts costs (up to 1.8x for 85+)
- Chronic conditions increase costs by ~50%
- Income affects out-of-pocket expenses
- Regional variations can be substantial

## Future Enhancements
1. Integration with real-time Medicare API data
2. More granular regional cost analysis
3. Machine learning for improved cost predictions
4. Additional plan type analysis
5. Provider network optimization
6. Prescription drug cost modeling

## Technical Details
- Python-based analysis
- Pandas for data processing
- NumPy for numerical computations
- Built on CMS public data APIs
- Modular design for easy updates

## Data Sources
All data is sourced from CMS public datasets:
- Medicare Geographic Variation
- Medicare Fee-for-Service Data
- Medicare Enrollment Data

## Disclaimer
This system provides estimates and recommendations based on historical Medicare data. Individual healthcare costs and optimal insurance choices may vary. Consult with healthcare providers and insurance professionals for personalized advice.
