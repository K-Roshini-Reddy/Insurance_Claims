from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field


class FraudRequest(BaseModel):
    claim_amount: float = Field(..., gt=0)
    num_prior_claims: int = Field(..., ge=0)
    days_since_policy_start: int = Field(..., ge=0)
    vin: Optional[str] = Field(default=None, min_length=11, max_length=17)


class FraudResponse(BaseModel):
    fraud_probability: float
    label: str
    threshold: float

    vin_status: str
    features_used: Dict[str, Any]

    degraded: bool
    confidence: Literal["HIGH", "LOW"]
    guardrail_flags: Dict[str, bool]
    guardrail_reasons: List[str]


class ModelInfoResponse(BaseModel):
    source: Literal["mlflow", "local"]
    loaded: bool

    model_uri: str
    model_name: Optional[str] = None
    stage: Optional[str] = None
    version: Optional[str] = None
    run_id: Optional[str] = None

    error: Optional[str] = None
