from flask import Blueprint, request, jsonify
from flasgger import swag_from
import os
import traceback
import uuid
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import io
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from app.utils.face_utils import extract_multiple_face_embeddings, find_best_match
from app.utils.upload_utils import request_group_face_vectors, request_presigned_url, upload_image_to_presigned_url, finalize_photo_upload, get_current_share_group_id

# 상수 정의
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
DEFAULT_CAMERA_TYPE = "back"
DEFAULT_DEVICE_ROTATION = "portraitUp"

# 로깅 설정
logger = logging.getLogger('upload_bp')
logger.setLevel(logging.ERROR)  # 에러 로그만 남김

upload_bp = Blueprint("debug_bp", __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@lru_cache(maxsize=100)
def rotate_image(image, angle):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if angle == 0:
        return image
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    return rotated

def rotate_image_90(image, quarters):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    quarters = quarters % 4
    if quarters == 0:
        return image
    
    if quarters == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif quarters == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif quarters == 3:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return image

class ImageProcessor:
    def __init__(self):
        self.cache = {}
    
    def process(self, image_bytes, camera_type, device_rotation):
        try:
            cache_key = f"{hash(image_bytes)}_{camera_type}_{device_rotation}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            image = Image.open(io.BytesIO(image_bytes))
            img_array = np.array(image)
            
            quarters = self._get_rotation_quarters(device_rotation)
            
            # 카메라 타입에 따른 처리
            if camera_type == "back":
                # 후면 카메라: 회전만 적용
                if quarters != 0:
                    img_array = rotate_image_90(img_array, quarters)
            else:  # front 카메라
                # 전면 카메라: 항상 좌우 반전 적용 후 회전
                img_array = cv2.flip(img_array, 1)  # 좌우 반전
                if quarters != 0:
                    img_array = rotate_image_90(img_array, quarters)
            
            result_image = Image.fromarray(img_array)
            if result_image.mode != 'RGB':
                result_image = result_image.convert('RGB')
            
            output_buffer = io.BytesIO()
            result_image.save(output_buffer, format="JPEG", quality=95)
            processed_bytes = output_buffer.getvalue()
            
            self.cache[cache_key] = processed_bytes
            return processed_bytes
            
        except Exception as e:
            logger.error(f"이미지 처리 오류: {str(e)}")
            return None
        finally:
            if 'image' in locals():
                del image
            if 'img_array' in locals():
                del img_array
    
    def _get_rotation_quarters(self, device_rotation):
        rotation_map = {
            "portraitUp": 0,
            "landscapeRight": 1,
            "portraitDown": 2,
            "landscapeLeft": 3
        }
        return rotation_map.get(device_rotation, 0)

image_processor = ImageProcessor()

@upload_bp.route("/upload", methods=["POST"])
@swag_from(os.path.join(os.path.dirname(__file__), "../../../docs/upload.yml"))
def upload_photo():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image part in the request"}), 400
            
        image_file = request.files["image"]
        if not allowed_file(image_file.filename):
            return jsonify({"error": "Invalid file type"}), 400
            
        original_image_bytes = image_file.read()

        share_group_id = request.form.get("shareGroupId", type=int)
        location = request.form.get("location", "").strip()
        take_at = request.form.get("takeAt", "").strip()
        
        camera_type = request.form.get("cameraType", DEFAULT_CAMERA_TYPE)
        device_rotation = request.form.get("deviceRotation", DEFAULT_DEVICE_ROTATION)
        
        if device_rotation.startswith("DeviceRotation."):
            device_rotation = device_rotation.split(".")[1]
            
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        access_token = auth_header.replace("Bearer ", "").strip()

        if not share_group_id:
            try:
                share_group_id = get_current_share_group_id(access_token)
            except Exception:
                return jsonify({"error": "Failed to fetch shareGroupId"}), 400
        
        processed_image_bytes = None
        image_bytes = original_image_bytes
                
        if camera_type and device_rotation:
            try:
                processed_image_bytes = image_processor.process(
                    original_image_bytes, 
                    camera_type, 
                    device_rotation
                )
                
                if processed_image_bytes:
                    if processed_image_bytes != original_image_bytes:
                        image_bytes = processed_image_bytes
            except Exception as e:
                logger.error(f"이미지 전처리 중 오류: {str(e)}")
                traceback.print_exc()

        with ThreadPoolExecutor(max_workers=3) as executor:
            # 얼굴 인식 처리
            result = extract_multiple_face_embeddings(image_bytes)
            valid_embeddings = [r for r in result if "embedding" in r]

            matched_profiles = []
            profile_id_set = set()
            if valid_embeddings:
                group_vector_response = request_group_face_vectors(share_group_id, access_token)
                for idx, embedding_info in enumerate(valid_embeddings):
                    query_embedding = embedding_info["embedding"]
                    best_matches = find_best_match(query_embedding, group_vector_response)
                    matched_profiles.append({
                        "faceIndex": idx,
                        "matches": best_matches
                    })
                    for match in best_matches:
                        profile_id_set.add(match["profileId"])
            profile_id_list = list(profile_id_set)

            # 이미지 업로드
            filename = str(uuid.uuid4()) + image_file.filename
            presigned_response = request_presigned_url([filename], access_token)
            upload_info = presigned_response["data"]["preSignedUrlInfoList"][0]
            photo_url = upload_info["photoUrl"]
            presigned_url = upload_info["preSignedUrl"]
            
            # S3에 이미지 업로드
            upload_future = executor.submit(
                upload_image_to_presigned_url, 
                presigned_url, 
                image_bytes
            )
            upload_future.result()  # 업로드 완료 대기
            
            # 서버에 업로드 완료 알림
            upload_result = finalize_photo_upload(
                share_group_id=share_group_id,
                photo_url=photo_url,
                profile_id_list=profile_id_list,
                location=location,
                taked_at=take_at,
                image_bytes=image_bytes,
                access_token=access_token
            )

        return jsonify({
            "message": "✅ 사진 업로드 및 등록 완료",
            "matchedProfiles": matched_profiles,
            "imageUrl": photo_url,
            "uploadResult": upload_result,
            "preprocessed": processed_image_bytes is not None and image_bytes == processed_image_bytes
        }), 200

    except Exception as e:
        logger.error(f"업로드 처리 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
