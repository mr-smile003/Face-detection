from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
from io import BytesIO
from app.services import face_encodings, find_matching_person
from typing import List, Dict, Optional
import numpy as np

# FastAPI app instance
app = FastAPI()

# Request model for input data
class ImageURLs(BaseModel):
    image_url: str

@app.post("/face_encodings/")
async def get_encodings(request: ImageURLs):
    # Get encodings for the single image URL
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
async def match_faces(request: FaceMatchRequest):
    try:
        # Convert all target encodings into numpy arrays
        targeted_encodings = [
            {"personId": target.personId, "encodings": np.array([float(e) for e in target.encodings])}
            for target in request.targetedEncodings
        ]
        
        # Convert and flatten the incoming encoding
        encoding_np = np.array([float(e) for e in request.encoding]).flatten()

        matching_person_id = find_matching_person(
            encoding=encoding_np,
            targeted_encodings=targeted_encodings,
            tolerance=0.5
        )
        
        return {"personId": matching_person_id}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))