"""
Unit tests for request/response validation. No dataset or model needed --
these test that Pydantic correctly accepts valid input and rejects
invalid input, which is exactly what protects /predict from bad data.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.append(str(Path(__file__).resolve().parent.parent / "app" / "api"))
from schemas import PredictionRequest


VALID_PAYLOAD = {
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
    "BILL_AMT5": 0, "BILL_AMT6": 0,
    "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0,
    "PAY_AMT5": 0, "PAY_AMT6": 0,
}


def test_valid_payload_is_accepted():
    request = PredictionRequest(**VALID_PAYLOAD)
    assert request.LIMIT_BAL == 20000
    assert request.AGE == 24


def test_missing_field_is_rejected():
    bad_payload = VALID_PAYLOAD.copy()
    del bad_payload["AGE"]
    with pytest.raises(ValidationError):
        PredictionRequest(**bad_payload)


def test_wrong_type_is_rejected():
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["AGE"] = "twenty-four"  # should be an int
    with pytest.raises(ValidationError):
        PredictionRequest(**bad_payload)
