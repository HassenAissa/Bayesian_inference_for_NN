import pytest
from src.example_main import return_val, return_one

def test_return_one_succeed():
    assert return_one() == 1

def test_return_one_fail():
    assert return_one() == 2

def test_return_val_succeed():
    assert return_val(10) == 10

def test_return_val_fail():
    assert return_val(15) == 4
