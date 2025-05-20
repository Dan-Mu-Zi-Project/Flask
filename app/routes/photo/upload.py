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

from app.utils.face_utils import extract_multiple_face_embeddings, find_best_match
from app.utils.upload_utils import request_group_face_vectors, request_presigned_url, upload_image_to_presigned_url, finalize_photo_upload, get_current_share_group_id

upload_bp = Blueprint("debug_bp", __name__)

# 이미지 회전 함수
def rotate_image(image, angle):
    """
    이미지를 주어진 각도로 회전합니다
    """
    # 이미지가 PIL Image인 경우 numpy 배열로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 이미지의 중심점
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 회전 행렬 계산
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 회전 적용
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    return rotated

# 이미지 전처리 함수
def preprocess_image(image_bytes, camera_type, device_rotation):
    """
    카메라 타입과 기기 방향에 따라 이미지를 전처리합니다
    """
    try:
        # 바이트 데이터로부터 이미지 읽기
        image = Image.open(io.BytesIO(image_bytes))
        
        # 이미지 형식과 크기 로깅
        print(f"원본 이미지 형식: {image.format}, 크기: {image.size}, 모드: {image.mode}")
        
        # OpenCV로 처리하기 위해 numpy 배열로 변환
        img_array = np.array(image)
        
        # 방향에 따른 회전 각도 설정
        rotation_angle = 0
        if device_rotation == "portraitUp":
            rotation_angle = 0
        elif device_rotation == "landscapeRight":
            rotation_angle = 90
        elif device_rotation == "portraitDown":
            rotation_angle = 180
        elif device_rotation == "landscapeLeft":
            rotation_angle = 270
        
        print(f"회전 각도: {rotation_angle}, 카메라 타입: {camera_type}")
        
        # 전면 카메라인 경우 좌우 반전 (셀피 카메라는 미러링되어 있기 때문)
        if camera_type == "front":
            img_array = cv2.flip(img_array, 1)  # 1은 좌우 반전
        
        # 회전 적용
        if rotation_angle != 0:
            img_array = rotate_image(img_array, rotation_angle)
        
        # 결과 이미지를 다시 PIL Image로 변환 후 바이트로 변환
        result_image = Image.fromarray(img_array)
        
        # RGB 모드로 변환 (RGBA나 다른 형식에서 변환이 필요할 수 있음)
        if result_image.mode != 'RGB':
            result_image = result_image.convert('RGB')
            print(f"이미지 모드 변환됨: {image.mode} -> RGB")
        
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format="JPEG", quality=95)
        
        processed_bytes = output_buffer.getvalue()
        print(f"전처리된 이미지 크기: {len(processed_bytes)} 바이트")
        
        return processed_bytes
    except Exception as e:
        print(f"이미지 전처리 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return None

@upload_bp.route("/upload", methods=["POST"])
@swag_from(os.path.join(os.path.dirname(__file__), "../../../docs/upload.yml"))
def upload_photo():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image part in the request"}), 400
        image_file = request.files["image"]
        image_bytes = image_file.read()

        share_group_id = request.form.get("shareGroupId", type=int)
        location = request.form.get("location", "").strip()
        take_at = request.form.get("takeAt", "").strip()
        
        # 카메라 타입과 기기 회전 방향 정보 가져오기
        camera_type = request.form.get("cameraType", "back")  # 기본값은 후면 카메라
        device_rotation = request.form.get("deviceRotation", "portraitUp")  # 기본값은 세로 정상

        # if not location:
        #     location = "None"
        # if not take_at:
        #     take_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        access_token = auth_header.replace("Bearer ", "").strip()

        if not share_group_id:
            try:
                share_group_id = get_current_share_group_id(access_token)
            except Exception:
                return jsonify({"error": "Failed to fetch shareGroupId"}), 400
                
        # 이미지 전처리 적용 (카메라 타입과 방향 정보가 있는 경우)
        if camera_type and device_rotation:
            try:
                processed_image_bytes = preprocess_image(image_bytes, camera_type, device_rotation)
                if processed_image_bytes:
                    print(f"이미지 전처리 완료: 카메라={camera_type}, 방향={device_rotation}")
                    image_bytes = processed_image_bytes
            except Exception as e:
                print(f"이미지 전처리 실패, 원본 이미지 사용: {str(e)}")
                traceback.print_exc()
                # 전처리 실패시 원본 이미지 사용 (이미 image_bytes에 있음)

        result = extract_multiple_face_embeddings(image_bytes)
        valid_embeddings = [r for r in result if "embedding" in r]
        # if not valid_embeddings:
        #     return jsonify({"error": "No valid face embeddings detected"}), 400

        query_embedding = valid_embeddings[0]["embedding"] if valid_embeddings else None
        
        # 얼굴 임베딩이 없으면 프로필 매칭 없이 진행
        profile_id_list = []
        if query_embedding:
            group_vector_response = request_group_face_vectors(share_group_id, access_token)
            best_matches = find_best_match(query_embedding, group_vector_response)
            profile_id_list = [match["profileId"] for match in best_matches]

        # if not profile_id_list:
        #     return jsonify({"error": "No similar profiles found"}), 404

        filename = str(uuid.uuid4()) + image_file.filename
        presigned_response = request_presigned_url([filename], access_token)
        upload_info = presigned_response["data"]["preSignedUrlInfoList"][0]
        photo_url = upload_info["photoUrl"]
        presigned_url = upload_info["preSignedUrl"]

        upload_image_to_presigned_url(presigned_url, image_bytes)

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
            "matchedProfiles": best_matches if 'best_matches' in locals() else [],
            "imageUrl": photo_url,
            "uploadResult": upload_result,
            "preprocessed": camera_type and device_rotation and processed_image_bytes is not None
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500