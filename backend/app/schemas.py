from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class FraudRequest(BaseModel):
    claim_amount: float = Field(..., ge=0)
    num_prior_claims: int = Field(..., ge=0)
    days_since_policy_start: int = Field(..., ge=0)
    vin: Optional[str] = Field(default=None, min_length=11, max_length=17)

class FraudResponse(BaseModel):
    fraud_probability: float
    label: str
    threshold: float
    vin_status: str
    features_used: Dict[str, Any]
