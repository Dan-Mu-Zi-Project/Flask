from deepface import DeepFace

import cv2
import numpy as np

from app.models.deepface_loader import FACE_DETECTION_MODEL, FACE_RECONGITION_MODEL

def extract_face_embedding(image_bytes):
    """
    여러 얼굴이 포함된 이미지에서 각 얼굴의 임베딩을 추출하는 함수.
    """
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image data.")

    try:
        result = DeepFace.represent(
            img_path=img,
            model_name=FACE_RECONGITION_MODEL,
            detector_backend=FACE_DETECTION_MODEL,
            enforce_detection=True
        )
    except Exception as e:
        raise ValueError(f"Face detection failed: {str(e)}")

    return result

def extract_multiple_face_embeddings(image_bytes):
    """
    여러 얼굴이 포함된 이미지에서 각 얼굴의 임베딩을 추출하는 함수.
    """
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image data.")

    try:
        face_objs = DeepFace.extract_faces(
            img_path=img,
            detector_backend=FACE_DETECTION_MODEL,
            enforce_detection=True,
            align=True,
            expand_percentage=0,
            grayscale=False,
            color_face="rgb",
            normalize_face=True,
            anti_spoofing=False,
        )
    except Exception as e:
        raise ValueError(f"Face extraction failed: {str(e)}")

    results = []

    for face_obj in face_objs:
        face_img = face_obj.get("face")
        facial_area = face_obj.get("facial_area", {})
        confidence = face_obj.get("confidence", None)

        try:
            embedding_result = extract_face_embedding(face_img)

            embedding = embedding_result[0]["embedding"]

            result = {
                "embedding": embedding,
                "facial_area": facial_area,
                "confidence": confidence
            }

            results.append(result)

        except Exception as e:
            # 개별 얼굴 처리 실패시
            results.append({
                "error": str(e),
                "facial_area": facial_area,
                "confidence": confidence
            })

    return results