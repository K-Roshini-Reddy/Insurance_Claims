from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

class VinAttributes(BaseModel):
    VIN: str = Field(min_length=17, max_length=17)
    Make: str = ""
    Model: str = ""
    ModelYear: Optional[int] = None
    BodyClass: str = ""
    VehicleType: str = ""
    FuelTypePrimary: str = ""
    EngineCylinders: Optional[float] = None
    DisplacementL: Optional[float] = None
    Manufacturer: str = ""
    PlantCountry: str = ""
    PlantState: str = ""

class VinIngestionMeta(BaseModel):
    source: str
    fetched_at_utc: str
    status: str  # OK | ERROR
    error_message: str = ""
