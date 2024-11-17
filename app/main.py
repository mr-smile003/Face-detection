from fastapi import FastAPI, HTTPException, Security, Header
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from pydantic import BaseModel, Field
import requests
from io import BytesIO
from app.services import face_encodings, find_matching_person
from typing import List, Dict, Optional
import numpy as np

# FastAPI app instance
app = FastAPI()

API_KEY_NAME = "x-api-key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header or api_key_header != 'GpTh18VYNq9F9Hsvrbpa7XOK92qKFU6C':
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid or missing API Key"
        )
    return api_key_header

# Request model for input data
class ImageURLs(BaseModel):
    image_url: str

@app.post("/face_encodings/")
async def get_encodings(request: ImageURLs, api_key: str = Security(get_api_key)):
    encodings = face_encodings(request.image_url)
    return {"encodings": encodings}


from pydantic import confloat

class TargetedEncoding(BaseModel):
    personId: str
    encodings: List[float]

class FaceMatchRequest(BaseModel):
    encoding: List[float]
    targetedEncodings: List[TargetedEncoding]
    tolerance: Optional[float] = Field(default=0.6, ge=0.0)

@app.post("/face_match/")
async def match_faces(request: FaceMatchRequest, api_key: str = Security(get_api_key)):
    try:
        targeted_encodings = [
            {"personId": target.personId, "encodings": np.array([float(e) for e in target.encodings])}
            for target in request.targetedEncodings
        ]
        
        encoding_np = np.array([float(e) for e in request.encoding]).flatten()

        matching_person_id = find_matching_person(
            encoding=encoding_np,
            targeted_encodings=targeted_encodings,
            tolerance=0.5
        )
        
        return {"personId": matching_person_id}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "message": "Face Matching Service is running"
    }