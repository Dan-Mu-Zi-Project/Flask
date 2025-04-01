from deepface import DeepFace
import cv2
import numpy as np
from scipy.spatial.distance import cosine

from app.models.deepface_loader import FACE_DETECTION_MODEL, FACE_RECONGITION_MODEL


def extract_face_embedding(image_input):
    """
    단일 얼굴 이미지에서 임베딩을 추출하는 함수.
    image_input: bytes 또는 np.ndarray (BGR or RGB)
    """
    if isinstance(image_input, bytes):
        image_array = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        detector_backend = FACE_DETECTION_MODEL
        enforce_detection = True
    elif isinstance(image_input, np.ndarray):
        img = image_input
        detector_backend = "skip"
        enforce_detection = False
    else:
        raise ValueError("Unsupported image input type")

    if img is None:
        raise ValueError("Invalid image data.")

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    try:
        result = DeepFace.represent(
            img_path=img,
            model_name=FACE_RECONGITION_MODEL,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection
        )
    except Exception as e:
        raise ValueError(f"Face embedding failed: {str(e)}")

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
        face_rgb = face_obj.get("face")
        facial_area = face_obj.get("facial_area", {})
        confidence = face_obj.get("confidence", None)

        try:
            embedding_result = extract_face_embedding(face_rgb)
            embedding = embedding_result[0]["embedding"]

            results.append({
                "embedding": embedding,
                "facial_area": facial_area,
                "confidence": confidence
            })

        except Exception as e:
            results.append({
                "error": str(e),
                "facial_area": facial_area,
                "confidence": confidence
            })

    return results


def find_best_match(query_vector, share_group_response):
    """
    query_vector: List[float]
    share_group_response: API 응답(JSON dict)
    """
    COSINE_THRESHOLD = 0.35

    best_matches = []

    members = share_group_response.get("data", {}).get("memberEmbeddingList", [])
    
    for member in members:
        member_id = member.get("memberId")
        profile_id = member.get("profileId")
        name = member.get("name")
        embedding_vectors = member.get("embeddingVectorList", [])

        min_distance = float("inf")
        best_match = None

        for item in embedding_vectors:
            vector_str = item.get("vector")
            if not vector_str:
                continue
            try:
                candidate_vector = json.loads(vector_str)
                distance = cosine(query_vector, candidate_vector)
                if distance < min_distance:
                    min_distance = distance
                    best_match = {
                        "memberId": member_id,
                        "profileId": profile_id,
                        "name": name,
                        "angleType": item.get("angleType"),
                        "distance": round(distance, 4)
                    }
            except Exception:
                continue

        if best_match and best_match["distance"] < COSINE_THRESHOLD:
            best_matches.append(best_match)

    # 거리 기준 오름차순 정렬
    best_matches.sort(key=lambda x: x["distance"])

    return best_matches