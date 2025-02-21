import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class InsurancePlan:
    plan_id: str
    name: str
    organization: str
    type: str
    monthly_premium: float
    health_deductible: float
    drug_deductible: float
    moop_amount: float
    drug_coverage: bool
    network_type: str
    star_rating: float
    primary_care_copay: str
    specialist_copay: str
    emergency_copay: str
    hospital_copay: str
    drug_coverage_gap: str
    dental_coverage: str
    vision_coverage: str
    hearing_coverage: str

@dataclass
class PlanMatch:
    plan: InsurancePlan
    match_score: float
    annual_cost_estimate: float
    network_adequacy: float
    coverage_score: float
    savings_potential: float
    pros: List[str]
    cons: List[str]

class PlanMatcher:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data/raw"
        self.ma_plans = None
        self.drug_plans = None
        self.load_plan_data()
        
    def load_plan_data(self, state: str = 'CA'):
        """Load insurance plan data"""
        try:
            # Load Medicare Advantage plans
            ma_path = self.data_dir / f"advantage_plans_{state}.csv"
            if ma_path.exists():
                self.ma_plans = pd.read_csv(ma_path)
                print(f"Loaded {len(self.ma_plans)} Medicare Advantage plans")
                
            # Load Part D drug plans
            drug_path = self.data_dir / f"drug_plans_{state}.csv"
            if drug_path.exists():
                self.drug_plans = pd.read_csv(drug_path)
                print(f"Loaded {len(self.drug_plans)} Part D plans")
                
        except Exception as e:
            print(f"Error loading plan data: {str(e)}")
            
    def _convert_to_plan_object(self, plan_data) -> InsurancePlan:
        """Convert DataFrame row to InsurancePlan object"""
        return InsurancePlan(
            plan_id=f"{plan_data['contract_id']}-{plan_data['plan_id']}",
            name=plan_data['plan_name'],
            organization=plan_data['organization_name'],
            type=plan_data['plan_type'],
            monthly_premium=float(plan_data['monthly_premium']),
            health_deductible=float(plan_data['health_deductible']),
            drug_deductible=float(plan_data['drug_deductible']),
            moop_amount=float(plan_data['moop_amount']),
            drug_coverage=bool(plan_data['drug_coverage']),
            network_type=plan_data['network_type'],
            star_rating=float(plan_data.get('star_rating', 3.0)),
            primary_care_copay=plan_data['primary_care_copay'],
            specialist_copay=plan_data['specialist_copay'],
            emergency_copay=plan_data['emergency_copay'],
            hospital_copay=plan_data['hospital_copay'],
            drug_coverage_gap=plan_data['drug_coverage_gap'],
            dental_coverage=plan_data['dental_coverage'],
            vision_coverage=plan_data['vision_coverage'],
            hearing_coverage=plan_data['hearing_coverage']
        )
        
    def _calculate_coverage_score(self, plan: InsurancePlan, 
                                health_profile: dict) -> float:
        """Calculate how well plan coverage matches health needs"""
        score = 0.0
        weights = {
            'drug_coverage': 0.3,
            'network_type': 0.2,
            'copays': 0.2,
            'extra_benefits': 0.1,
            'star_rating': 0.2
        }
        
        # Drug coverage
        if health_profile.get('needs_prescriptions', False):
            score += weights['drug_coverage'] * (1.0 if plan.drug_coverage else 0.0)
            
        # Network type
        if health_profile.get('prefers_ppo', False):
            score += weights['network_type'] * (1.0 if 'PPO' in plan.type else 0.5)
            
        # Copays (convert $ amounts to floats)
        def extract_amount(copay: str) -> float:
            try:
                return float(copay.replace('$', ''))
            except:
                return 50.0  # Default high copay
                
        copay_score = 1.0 - min(1.0, (
            extract_amount(plan.primary_care_copay) +
            extract_amount(plan.specialist_copay)
        ) / 200.0)  # Normalize to 0-1
        score += weights['copays'] * copay_score
        
        # Extra benefits
        if health_profile.get('age', 0) >= 65:
            extra_benefits_score = sum([
                1 if plan.dental_coverage == 'Yes' else 0,
                1 if plan.vision_coverage == 'Yes' else 0,
                1 if plan.hearing_coverage == 'Yes' else 0
            ]) / 3.0
            score += weights['extra_benefits'] * extra_benefits_score
            
        # Star rating
        score += weights['star_rating'] * (plan.star_rating / 5.0)
        
        return score
        
    def _calculate_cost_score(self, plan: InsurancePlan, 
                            health_profile: dict) -> Tuple[float, float]:
        """Calculate cost score and annual cost estimate"""
        # Estimate annual costs
        annual_premium = plan.monthly_premium * 12
        estimated_medical_costs = health_profile.get('estimated_annual_costs', 5000)
        
        # Calculate out of pocket costs
        if estimated_medical_costs <= plan.health_deductible:
            out_of_pocket = estimated_medical_costs
        else:
            covered_amount = estimated_medical_costs - plan.health_deductible
            out_of_pocket = plan.health_deductible + (covered_amount * 0.2)  # Assume 20% coinsurance
            
        # Cap at MOOP
        out_of_pocket = min(out_of_pocket, plan.moop_amount)
        
        # Add drug costs if needed
        if health_profile.get('needs_prescriptions', False) and plan.drug_coverage:
            out_of_pocket += plan.drug_deductible
            
        total_cost = annual_premium + out_of_pocket
        
        # Calculate cost score (lower is better)
        max_acceptable = health_profile.get('max_annual_budget', 10000)
        cost_score = 1.0 - min(1.0, total_cost / max_acceptable)
        
        return cost_score, total_cost
        
    def _analyze_plan_fit(self, plan: InsurancePlan, health_profile: dict,
                         cost_estimate: float) -> Tuple[List[str], List[str]]:
        """Analyze pros and cons of a plan"""
        pros = []
        cons = []
        
        # Analyze premium
        if plan.monthly_premium <= health_profile.get('max_monthly_premium', 100):
            pros.append(f"Low monthly premium (${plan.monthly_premium:.2f})")
        else:
            cons.append(f"High monthly premium (${plan.monthly_premium:.2f})")
            
        # Analyze deductible
        if plan.health_deductible == 0:
            pros.append("No health deductible")
        elif plan.health_deductible <= 500:
            pros.append(f"Low health deductible (${plan.health_deductible:.2f})")
        else:
            cons.append(f"High health deductible (${plan.health_deductible:.2f})")
            
        # Analyze network
        if 'PPO' in plan.type:
            pros.append("Flexible provider choice with PPO network")
        else:
            cons.append("Limited to HMO network providers")
            
        # Analyze drug coverage
        if health_profile.get('needs_prescriptions', False):
            if plan.drug_coverage:
                pros.append("Includes prescription drug coverage")
                if plan.drug_coverage_gap == 'Yes':
                    pros.append("Includes coverage in the drug gap")
            else:
                cons.append("No prescription drug coverage")
                
        # Analyze extra benefits
        if health_profile.get('age', 0) >= 65:
            extra_benefits = []
            if plan.dental_coverage == 'Yes':
                extra_benefits.append("dental")
            if plan.vision_coverage == 'Yes':
                extra_benefits.append("vision")
            if plan.hearing_coverage == 'Yes':
                extra_benefits.append("hearing")
            if extra_benefits:
                pros.append(f"Includes {', '.join(extra_benefits)} coverage")
            else:
                cons.append("No extra benefits included")
                
        # Analyze star rating
        if plan.star_rating >= 4.0:
            pros.append(f"High quality rating ({plan.star_rating} stars)")
        elif plan.star_rating < 3.0:
            cons.append(f"Low quality rating ({plan.star_rating} stars)")
            
        # Analyze cost
        annual_cost = cost_estimate
        budget = health_profile.get('max_annual_budget', 10000)
        if annual_cost <= budget * 0.8:
            pros.append(f"Estimated annual cost (${annual_cost:.2f}) well within budget")
        elif annual_cost <= budget:
            pros.append(f"Estimated annual cost (${annual_cost:.2f}) within budget")
        else:
            cons.append(f"Estimated annual cost (${annual_cost:.2f}) exceeds budget")
            
        return pros, cons
        
    def find_matching_plans(self, health_profile: dict) -> List[PlanMatch]:
        """Find and rank matching insurance plans"""
        if self.ma_plans is None:
            print("No plan data available")
            return []
            
        matches = []
        
        for _, plan_data in self.ma_plans.iterrows():
            # Convert to plan object
            plan = self._convert_to_plan_object(plan_data)
            
            # Calculate coverage score
            coverage_score = self._calculate_coverage_score(plan, health_profile)
            
            # Calculate cost score and estimate
            cost_score, annual_cost = self._calculate_cost_score(plan, health_profile)
            
            # Calculate network adequacy
            network_score = 0.9 if 'PPO' in plan.type else 0.7
            
            # Calculate overall match score
            match_score = (
                coverage_score * 0.4 +
                cost_score * 0.4 +
                network_score * 0.2
            )
            
            # Calculate potential savings
            avg_cost = health_profile.get('estimated_annual_costs', 5000)
            savings = max(0, avg_cost - annual_cost)
            
            # Analyze pros and cons
            pros, cons = self._analyze_plan_fit(plan, health_profile, annual_cost)
            
            # Create match object
            match = PlanMatch(
                plan=plan,
                match_score=match_score,
                annual_cost_estimate=annual_cost,
                network_adequacy=network_score,
                coverage_score=coverage_score,
                savings_potential=savings,
                pros=pros,
                cons=cons
            )
            
            matches.append(match)
            
        # Sort by match score
        matches.sort(key=lambda x: x.match_score, reverse=True)
        
        return matches[:5]  # Return top 5 matches

def main():
    """Test the plan matcher"""
    matcher = PlanMatcher()
    
    # Test health profile
    health_profile = {
        'age': 67,
        'estimated_annual_costs': 5000,
        'max_annual_budget': 8000,
        'max_monthly_premium': 100,
        'needs_prescriptions': True,
        'prefers_ppo': True,
        'risk_level': 'medium'
    }
    
    print("\nFinding matching plans...")
    print("=========================")
    
    print("\nHealth Profile:")
    for key, value in health_profile.items():
        print(f"{key}: {value}")
    
    # Find matching plans
    matches = matcher.find_matching_plans(health_profile)
    
    # Print results
    print("\nTop Plan Matches:")
    print("================")
    
    for i, match in enumerate(matches, 1):
        print(f"\n{i}. {match.plan.name} ({match.plan.type})")
        print(f"Organization: {match.plan.organization}")
        print(f"Match Score: {match.match_score:.1%}")
        print(f"Monthly Premium: ${match.plan.monthly_premium:.2f}")
        print(f"Annual Cost Estimate: ${match.annual_cost_estimate:,.2f}")
        print(f"Network Adequacy: {match.network_adequacy:.1%}")
        print(f"Coverage Score: {match.coverage_score:.1%}")
        print(f"Potential Savings: ${match.savings_potential:,.2f}")
        
        print("\nKey Features:")
        print(f"- Health Deductible: ${match.plan.health_deductible:,.2f}")
        print(f"- Drug Coverage: {'Yes' if match.plan.drug_coverage else 'No'}")
        print(f"- Primary Care Copay: {match.plan.primary_care_copay}")
        print(f"- Specialist Copay: {match.plan.specialist_copay}")
        print(f"- Star Rating: {match.plan.star_rating}")
        
        print("\nPros:")
        for pro in match.pros:
            print(f"+ {pro}")
            
        print("\nCons:")
        for con in match.cons:
            print(f"- {con}")

if __name__ == "__main__":
    main()
