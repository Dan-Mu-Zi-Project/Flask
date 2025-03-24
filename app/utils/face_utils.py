from deepface import DeepFace
import cv2
import numpy as np

from app.models.deepface_loader import FACE_DETECTION_MODEL, FACE_RECONGITION_MODEL

def extract_face_embedding(image_bytes):
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image data.")

    result = DeepFace.represent(
        img_path=img,
        model_name=FACE_RECONGITION_MODEL,
        detector_backend=FACE_DETECTION_MODEL,
        enforce_detection=True
    )

    return result[0]
