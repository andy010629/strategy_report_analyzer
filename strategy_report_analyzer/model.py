from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel


class MT4StrategyReportModel(BaseModel):
    name: str =""