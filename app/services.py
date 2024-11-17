import face_recognition
import requests
from io import BytesIO
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
from scipy.spatial.distance import cdist

def face_encodings(image_url: str):
    try:
        # Download the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()
        
        # Convert the image to a format that face_recognition can use
        img_array = face_recognition.load_image_file(BytesIO(response.content))
        
        # Get face encodings for all detected faces
        encodings = face_recognition.face_encodings(img_array)
        if not encodings:
            print("No faces detected in the image.")
            return []
        
        # Convert numpy arrays to lists for JSON serialization
        return [encoding.tolist() for encoding in encodings]
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")


def find_matching_person(encoding, targeted_encodings, tolerance):
    # Extract the encodings and person IDs
    target_ids = [target["personId"] for target in targeted_encodings]
    target_vectors = np.array([target["encodings"] for target in targeted_encodings])
    
    # Calculate distances between the encoding and all targets
    distances = cdist([encoding], target_vectors, metric="euclidean").flatten()
    
    # Find the closest match within the tolerance
    min_distance = distances.min()
    if min_distance <= tolerance:
        return target_ids[np.argmin(distances)]
    return None