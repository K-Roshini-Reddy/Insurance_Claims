from pydantic import BaseModel, Field

class FraudRequest(BaseModel):
    claim_amount: float = Field(..., ge=0)
    num_prior_claims: int = Field(..., ge=0)
    days_since_policy_start: int = Field(..., ge=0)

class FraudResponse(BaseModel):
    fraud_probability: float
    label: str
    threshold: float
