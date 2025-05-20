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
import logging.handlers

from app.utils.face_utils import extract_multiple_face_embeddings, find_best_match
from app.utils.upload_utils import request_group_face_vectors, request_presigned_url, upload_image_to_presigned_url, finalize_photo_upload, get_current_share_group_id

# 로깅 설정
logger = logging.getLogger('upload_bp')
logger.setLevel(logging.DEBUG)

# 로그 파일 핸들러 설정 (10MB 크기로 로테이션, 최대 5개 파일 유지)
log_file = 'upload_debug.log'
handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 콘솔 출력 핸들러 추가 (터미널에서도 로그 확인 가능)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("==== 로그 시스템 초기화 완료 ====")

upload_bp = Blueprint("debug_bp", __name__)

# 이미지 회전 함수
def rotate_image(image, angle):
    """
    이미지를 주어진 각도로 회전합니다 (잘림 없이)
    """
    # 이미지가 PIL Image인 경우 numpy 배열로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 각도가 0이면 회전하지 않음
    if angle == 0:
        return image
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 회전 행렬 계산
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 회전 후 bounding box 크기 계산
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # 회전 행렬의 이동 보정
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # 회전 적용 (새로운 크기로)
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    
    return rotated

# 직접 90도 단위로 이미지를 회전시키는 함수 추가
def rotate_image_90(image, quarters):
    """
    이미지를 90도의 배수로 회전시키는 함수
    
    Args:
        image: 회전할 이미지 (NumPy 배열)
        quarters: 90도 회전 횟수 (1: 90도, 2: 180도, 3: 270도)
    
    Returns:
        회전된 이미지
    """
    # 이미지가 PIL Image인 경우 numpy 배열로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # quarters를 0~3 사이 값으로 보정
    quarters = quarters % 4
    if quarters == 0:
        return image
    
    # 90도 단위로 회전 (cv2.rotate 사용)
    if quarters == 1:  # 90도
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif quarters == 2:  # 180도
        return cv2.rotate(image, cv2.ROTATE_180)
    elif quarters == 3:  # 270도
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return image

# 이미지 전처리 함수
def preprocess_image(image_bytes, camera_type, device_rotation):
    """
    카메라 타입과 기기 방향에 따라 이미지를 전처리합니다
    """
    try:
        # 바이트 데이터로부터 이미지 읽기
        image = Image.open(io.BytesIO(image_bytes))
        logger.info(f"원본 이미지 크기: {image.size}, 모드: {image.mode}")
        
        # OpenCV로 처리하기 위해 numpy 배열로 변환
        img_array = np.array(image)
        
        # 방향에 따른 회전 각도 설정
        quarters = 0  # 90도 회전 횟수
        if device_rotation == "portraitUp":
            quarters = 0
        elif device_rotation == "landscapeRight":
            quarters = 1  # 90도 시계방향
        elif device_rotation == "portraitDown":
            quarters = 2  # 180도
        elif device_rotation == "landscapeLeft":
            quarters = 3  # 270도 시계방향 (90도 반시계방향)
        
        logger.info(f"이미지 처리 시작: 방향={device_rotation}, 회전={quarters*90}도, 카메라={camera_type}")
        
        # 변환이 필요한지 여부
        needs_processing = quarters != 0 or camera_type == "front"
        if not needs_processing:
            logger.info("이미지 변환 불필요: 후면 카메라/세로 방향")
            return image_bytes  # 원본 반환
        
        # 이미지 처리 전 크기 저장
        before_shape = img_array.shape
        
        # 최적화된 이미지 처리 로직
        if camera_type == "back":
            # 후면 카메라: 단순히 회전만 적용
            if quarters != 0:
                logger.info(f"후면 카메라 회전 적용: {quarters*90}도")
                img_array = rotate_image_90(img_array, quarters)
        else:
            # 전면 카메라: 최적화된 회전 방향 적용
            if device_rotation == "portraitUp":
                # 세로 정방향: 좌우 반전만 적용
                logger.info("전면 카메라 세로 정방향: 좌우 반전만 적용")
                img_array = cv2.flip(img_array, 1)  # 좌우 반전
            
            elif device_rotation == "portraitDown":
                # 세로 뒤집힘: 좌우 반전 + 180도 회전
                logger.info("전면 카메라 세로 뒤집힘: 좌우 반전 + 180도 회전")
                img_array = cv2.flip(img_array, 1)  # 좌우 반전
                img_array = rotate_image_90(img_array, 2)  # 180도 회전
            
            elif device_rotation == "landscapeRight":
                # 오른쪽으로 눕힘: 90도 회전 + 좌우 반전만 적용
                # (90도 회전 후 좌우 반전이 가장 효율적)
                logger.info("전면 카메라 오른쪽 방향: 90도 회전 + 좌우 반전")
                img_array = rotate_image_90(img_array, 1)  # 90도 회전
                img_array = cv2.flip(img_array, 1)  # 좌우 반전
            
            elif device_rotation == "landscapeLeft":
                # 왼쪽으로 눕힘: 90도 반시계 회전 + 좌우 반전
                # (90도 반시계 회전은 cv2.ROTATE_90_COUNTERCLOCKWISE 직접 적용)
                logger.info("전면 카메라 왼쪽 방향: 90도 반시계 회전 + 좌우 반전")
                img_array = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 90도 반시계 회전
                img_array = cv2.flip(img_array, 1)  # 좌우 반전
        
        # 이미지 처리 후 크기 확인
        after_shape = img_array.shape
        logger.info(f"이미지 변환 결과: {before_shape} -> {after_shape}")
        
        # 결과 이미지를 다시 PIL Image로 변환 후 바이트로 변환
        result_image = Image.fromarray(img_array)
        
        # RGB 모드로 변환 (RGBA나 다른 형식에서 변환이 필요할 수 있음)
        if result_image.mode != 'RGB':
            result_image = result_image.convert('RGB')
            logger.info(f"이미지 모드 변환: {image.mode} -> RGB")
        
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format="JPEG", quality=95)
        processed_bytes = output_buffer.getvalue()
        
        # 처리 성공 여부 확인
        process_success = len(processed_bytes) != len(image_bytes) or processed_bytes != image_bytes
        logger.info(f"이미지 처리 완료: 원본 크기={len(image_bytes)}, 처리 후={len(processed_bytes)}, 변경됨={process_success}")
        
        # 디버깅용: 처리된 이미지 저장
        try:
            debug_filename = f"debug_processed_{camera_type}_{device_rotation}.jpg"
            with open(debug_filename, "wb") as f:
                f.write(processed_bytes)
            logger.info(f"디버깅: 처리된 이미지 저장됨 ({debug_filename})")
        except Exception as e:
            logger.error(f"이미지 저장 실패: {e}")
        
        return processed_bytes
    except Exception as e:
        logger.error(f"이미지 전처리 오류: {str(e)}")
        traceback.print_exc()
        return None

@upload_bp.route("/upload", methods=["POST"])
@swag_from(os.path.join(os.path.dirname(__file__), "../../../docs/upload.yml"))
def upload_photo():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image part in the request"}), 400
        image_file = request.files["image"]
        original_image_bytes = image_file.read()
        
        # 원본 이미지 크기 로깅
        logger.info(f"요청 수신: 원본 이미지 크기={len(original_image_bytes)} 바이트")

        share_group_id = request.form.get("shareGroupId", type=int)
        location = request.form.get("location", "").strip()
        take_at = request.form.get("takeAt", "").strip()
        
        # 카메라 타입과 기기 회전 방향 정보 가져오기
        camera_type = request.form.get("cameraType", "back")  # 기본값은 후면 카메라
        device_rotation = request.form.get("deviceRotation", "portraitUp")  # 기본값은 세로 정상
        
        # DeviceRotation.landscapeLeft와 같은 형식에서 실제 방향 문자열 추출
        if device_rotation.startswith("DeviceRotation."):
            device_rotation = device_rotation.split(".")[1]  # Enum prefix 제거
            
        logger.info(f"이미지 메타데이터: 카메라={camera_type}, 방향={device_rotation}")

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        access_token = auth_header.replace("Bearer ", "").strip()

        if not share_group_id:
            try:
                share_group_id = get_current_share_group_id(access_token)
            except Exception:
                return jsonify({"error": "Failed to fetch shareGroupId"}), 400
        
        # 처리될 이미지 바이트를 원본으로 초기화
        processed_image_bytes = None
        image_bytes = original_image_bytes
                
        # 이미지 전처리 적용 (카메라 타입과 방향 정보가 있는 경우)
        if camera_type and device_rotation:
            try:
                logger.info(f"이미지 전처리 시작...")
                processed_image_bytes = preprocess_image(original_image_bytes, camera_type, device_rotation)
                
                if processed_image_bytes:
                    logger.info(f"전처리 성공: 처리 전={len(original_image_bytes)}바이트, 처리 후={len(processed_image_bytes)}바이트")
                    # 원본과 처리된 이미지가 다른지 확인
                    is_different = original_image_bytes != processed_image_bytes
                    logger.info(f"이미지 변경 확인: {'변경됨' if is_different else '원본과 동일 (변경 안됨)'}")
                    
                    # 전처리된 이미지로 확실히 교체
                    if is_different:
                        image_bytes = processed_image_bytes
                        logger.info("✅ 전처리된 이미지로 교체 완료")
                    else:
                        logger.warning("⚠️ 전처리 후에도 이미지 변경 없음")
                else:
                    logger.warning("⚠️ 전처리 실패, 원본 이미지 사용")
            except Exception as e:
                logger.error(f"이미지 전처리 중 오류: {str(e)}")
                traceback.print_exc()

        # 얼굴 인식 처리
        logger.info(f"얼굴 인식 시작: 이미지 크기={len(image_bytes)}바이트")
        result = extract_multiple_face_embeddings(image_bytes)
        valid_embeddings = [r for r in result if "embedding" in r]
        logger.info(f"얼굴 인식 결과: {len(valid_embeddings)}개 얼굴 감지됨")
        
        # 얼굴 임베딩이 없으면 프로필 매칭 없이 진행
        profile_id_list = []
        if valid_embeddings:
            query_embedding = valid_embeddings[0]["embedding"]
            group_vector_response = request_group_face_vectors(share_group_id, access_token)
            best_matches = find_best_match(query_embedding, group_vector_response)
            profile_id_list = [match["profileId"] for match in best_matches]
            logger.info(f"프로필 매칭 결과: {len(profile_id_list)}개 프로필 매치됨")

        # 이미지 업로드
        filename = str(uuid.uuid4()) + image_file.filename
        presigned_response = request_presigned_url([filename], access_token)
        upload_info = presigned_response["data"]["preSignedUrlInfoList"][0]
        photo_url = upload_info["photoUrl"]
        presigned_url = upload_info["preSignedUrl"]
        
        # 업로드 직전 이미지 확인
        logger.info(f"S3 업로드 직전: 이미지 크기={len(image_bytes)}바이트")
        # 원본과 다른지 확인
        is_original = image_bytes == original_image_bytes
        logger.info(f"업로드 이미지 상태: {'원본 이미지임' if is_original else '처리된 이미지임'}")
        
        # 디버깅용 업로드 전 이미지 저장
        try:
            debug_upload_filename = f"debug_upload_{camera_type}_{device_rotation}.jpg"
            with open(debug_upload_filename, "wb") as f:
                f.write(image_bytes)
            logger.info(f"업로드 직전 이미지 저장: {debug_upload_filename}")
        except Exception as e:
            logger.error(f"디버그 이미지 저장 실패: {e}")
        
        # S3에 이미지 업로드
        logger.info(f"S3 업로드 시작: 이미지 크기={len(image_bytes)}바이트")
        upload_image_to_presigned_url(presigned_url, image_bytes)
        logger.info(f"S3 업로드 완료: URL={photo_url}")
        
        # 서버에 업로드 완료 알림
        logger.info(f"업로드 완료 요청 시작...")
        upload_result = finalize_photo_upload(
            share_group_id=share_group_id,
            photo_url=photo_url,
            profile_id_list=profile_id_list,
            location=location,
            taked_at=take_at,
            image_bytes=image_bytes,
            access_token=access_token
        )
        logger.info(f"업로드 완료 요청 성공")

        return jsonify({
            "message": "✅ 사진 업로드 및 등록 완료",
            "matchedProfiles": best_matches if 'best_matches' in locals() else [],
            "imageUrl": photo_url,
            "uploadResult": upload_result,
            "preprocessed": processed_image_bytes is not None and image_bytes == processed_image_bytes
        }), 200

    except Exception as e:
        logger.error(f"업로드 처리 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
