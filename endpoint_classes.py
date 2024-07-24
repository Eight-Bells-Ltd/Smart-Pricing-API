from pydantic import BaseModel
from typing import List

class Service(BaseModel):
    provider_id: str
    consumer_id: str
    minprice: float
    maxprice: float
    service_id: str

class ServicesPayload(BaseModel):
    services: List[Service]