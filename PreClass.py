#Format input
#1. Librairie
from pydantic import BaseModel

# 2. Classe des predicteurs
class PreClass(BaseModel):
    EXT_SOURCE_3: float 
    EXT_SOURCE_2: float 
    DAYS_REGISTRATION: float
    DAYS_BIRTH: float
    PAYMENT_RATE: float
    DAYS_ID_PUBLISH : float
    DAYS_LAST_PHONE_CHANGE : float
    REGION_POPULATION_RELATIVE : float