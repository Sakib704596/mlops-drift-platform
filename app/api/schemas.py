"""
Pydantic schemas for the credit-default prediction API.
These field names match the cleaned columns from training/data.py exactly.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
                "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
                "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
                "BILL_AMT5": 0, "BILL_AMT6": 0,
                "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0,
                "PAY_AMT5": 0, "PAY_AMT6": 0,
            }
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = no default, 1 = default")
    probability: float = Field(..., description="Predicted probability of default")
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_stage: str
