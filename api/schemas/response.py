from pydantic import BaseModel

class DetectionResponse(BaseModel):
    classification: str
    confidence: float
    explanation: str
