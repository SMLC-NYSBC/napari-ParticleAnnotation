from typing import List

from pydantic import BaseModel


class Consensus(BaseModel):
    corrected_particle: List[List[float]]


class String(BaseModel):
    name_: str


class InitialValues(BaseModel):
    m_name: str
    f_name: str
    n_part: int

