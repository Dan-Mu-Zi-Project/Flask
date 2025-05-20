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
    이미지를 주어진 각도로 회전합니다 (잘림 없이)
    """
    # 이미지가 PIL Image인 경우 numpy 배열로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    print(f"회전 시작: 이미지 크기 {image.shape}, 각도 {angle}")
    
    # 각도가 0이면 회전하지 않음
    if angle == 0:
        print("회전 각도가 0도여서 회전 생략")
        return image
        
    # OpenCV는 이미지를 반시계 방향으로 회전시키므로 양수 각도를 사용
    cv_angle = angle # OpenCV는 반시계 방향이 양수

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    print(f"회전 중심: {center}, 높이: {h}, 너비: {w}")

    # 회전 행렬 계산
    rotation_matrix = cv2.getRotationMatrix2D(center, cv_angle, 1.0)
    
    print(f"회전 행렬: \n{rotation_matrix}")

    # 회전 후 bounding box 크기 계산
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    print(f"새 이미지 크기 계산: 너비 {new_w}, 높이 {new_h}")

    # 회전 행렬의 이동 보정
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    print(f"이동 보정된 회전 행렬: \n{rotation_matrix}")

    # 회전 적용 (새로운 크기로)
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    
    print(f"회전 완료: 결과 이미지 크기 {rotated.shape}")
    
    # 회전이 제대로 적용되었는지 확인
    if rotated.shape[:2] == image.shape[:2] and angle != 0:
        print("⚠️ 경고: 회전 후에도 이미지 크기가 변경되지 않았습니다!")
    
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
    
    print(f"90도 단위 회전: {quarters * 90}도")
    
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
        
        # 이미지 형식과 크기 로깅
        print(f"원본 이미지 형식: {image.format}, 크기: {image.size}, 모드: {image.mode}")
        
        # OpenCV로 처리하기 위해 numpy 배열로 변환
        img_array = np.array(image)
        original_array = img_array.copy()  # 원본 이미지 배열 복사
        
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
        
        rotation_angle = quarters * 90
        print(f"회전 각도: {rotation_angle}도 ({quarters} 쿼터), 카메라 타입: {camera_type}")
        
        # 변환이 필요한지 여부
        needs_processing = quarters != 0 or camera_type == "front"
        if not needs_processing:
            print("이미지 변환 필요 없음: 후면 카메라, 세로 방향")
            return image_bytes  # 원본 반환
        
        # 카메라 타입과 방향에 따른 최적의 이미지 처리
        if camera_type == "back":
            # 후면 카메라: 단순히 회전만 적용
            if quarters != 0:
                print(f"후면 카메라 회전 적용: {quarters} 쿼터 (90도 단위)")
                # 새 rotate_image_90 함수 사용
                img_array = rotate_image_90(img_array, quarters)
        else:
            # 전면 카메라: 방향에 따라 다르게 처리
            if device_rotation == "portraitUp" or device_rotation == "portraitDown":
                # 세로 방향: 먼저 좌우 반전 후 회전
                print("전면 카메라 세로 방향: 좌우 반전 후 회전")
                img_array = cv2.flip(img_array, 1)  # 좌우 반전
                if quarters != 0:
                    # 새 rotate_image_90 함수 사용
                    img_array = rotate_image_90(img_array, quarters)
            elif device_rotation == "landscapeRight":
                # 오른쪽으로 회전: 90도 회전 후 필요시 반전
                print("전면 카메라 오른쪽 방향: 90도 회전 (1쿼터)")
                img_array = rotate_image_90(img_array, 1)  # 90도 시계방향
                # 필요시 추가 반전
                img_array = cv2.flip(img_array, 1)  # 좌우 반전
            elif device_rotation == "landscapeLeft":
                # 왼쪽으로 회전: 270도 회전 후 필요시 반전
                print("전면 카메라 왼쪽 방향: 270도 회전 (3쿼터)")
                img_array = rotate_image_90(img_array, 3)  # 270도 시계방향
                # 필요시 추가 반전
                img_array = cv2.flip(img_array, 1)  # 좌우 반전
        
        # 이미지가 실제로 변경되었는지 확인
        is_modified = not np.array_equal(original_array, img_array)
        print(f"이미지 변경 확인: {'변경됨' if is_modified else '변경 안됨!!!'}")
        
        if not is_modified:
            print("⚠️ 경고: 전처리 후에도 이미지가 변경되지 않았습니다!")
            # 강제로 이미지 회전 시도
            print("강제 90도 회전 시도")
            img_array = rotate_image_90(img_array, 1)  # 강제로 90도 회전
        
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
        
        # 원본과 처리 후 이미지 크기가 같은지 확인
        if len(processed_bytes) == len(image_bytes):
            print("⚠️ 경고: 원본과 처리 후 이미지 크기가 동일합니다!")
        
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
        original_image_bytes = image_bytes  # 원본 이미지 저장
        if camera_type and device_rotation:
            try:
                # 원본 이미지 크기 기록
                print(f"원본 이미지 크기: {len(image_bytes)} 바이트")
                
                processed_image_bytes = preprocess_image(image_bytes, camera_type, device_rotation)
                if processed_image_bytes:
                    print(f"이미지 전처리 완료: 카메라={camera_type}, 방향={device_rotation}")
                    print(f"처리 후 이미지 크기: {len(processed_image_bytes)} 바이트")
                    
                    # 원본과 처리된 이미지가 다른지 확인
                    is_different = original_image_bytes != processed_image_bytes
                    print(f"원본과 처리된 이미지 비교: {'다름' if is_different else '동일함!!!'}")
                    
                    if not is_different:
                        print("⚠️ 경고: 전처리 후에도 원본 이미지와 동일합니다!")
                    
                    # 디버깅용으로 전처리된 이미지 저장 (서버에 쓰기 권한이 있어야 함)
                    try:
                        # 원본 이미지 저장
                        original_filename = f"debug_original_{camera_type}_{device_rotation}.jpg"
                        with open(original_filename, "wb") as f:
                            f.write(original_image_bytes)
                        print(f"원본 이미지 저장됨: {original_filename}")
                        
                        # 처리된 이미지 저장
                        debug_filename = f"debug_processed_{camera_type}_{device_rotation}.jpg"
                        with open(debug_filename, "wb") as f:
                            f.write(processed_image_bytes)
                        print(f"처리된 이미지 저장됨: {debug_filename}")
                    except Exception as save_err:
                        print(f"디버깅 이미지 저장 실패: {str(save_err)}")
                    
                    # 전처리된 이미지로 명시적으로 교체
                    image_bytes = processed_image_bytes
                    print("✅ 전처리된 이미지로 성공적으로 교체됨")
            except Exception as e:
                print(f"이미지 전처리 실패, 원본 이미지 사용: {str(e)}")
                traceback.print_exc()
                # 전처리 실패시 원본 이미지 사용 (이미 image_bytes에 있음)

        # 이미지 바이트 유효성 검증
        if len(image_bytes) == 0:
            return jsonify({"error": "Invalid image data (zero length)"}), 400
        
        # 이미지가 원본과 다른지 최종 확인
        is_original = image_bytes == original_image_bytes
        print(f"최종 상태: {'원본 이미지 사용' if is_original else '전처리된 이미지 사용'}")
            
        print(f"얼굴 인식에 사용할 이미지 크기: {len(image_bytes)} 바이트")

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

        print(f"S3 업로드 시작: URL={photo_url}, 이미지 크기={len(image_bytes)} 바이트")
        
        # 업로드 전 이미지가 원본과 다른지 다시 확인
        is_still_original = image_bytes == original_image_bytes
        if is_still_original:
            print("⚠️ 경고: S3 업로드 직전에도 여전히 원본 이미지가 사용됩니다!")
            # 마지막 시도로 다시 강제 처리
            try:
                print("마지막 시도: 이미지 강제 90도 회전 시도")
                temp_img = Image.open(io.BytesIO(image_bytes))
                temp_array = np.array(temp_img)
                
                # 강제로 90도 회전 (OpenCV 직접 사용)
                # 미세 회전이 아닌 명확한 90도 회전 사용
                rotated_array = cv2.rotate(temp_array, cv2.ROTATE_90_CLOCKWISE)
                
                # 다시 이미지로 변환
                result_img = Image.fromarray(rotated_array)
                output = io.BytesIO()
                result_img.save(output, format="JPEG", quality=95)
                image_bytes = output.getvalue()
                
                print(f"강제 90도 회전 후 이미지 크기: {len(image_bytes)} 바이트")
                
                # 강제 처리된 이미지 저장
                forced_filename = f"debug_forced_{camera_type}_{device_rotation}.jpg"
                with open(forced_filename, "wb") as f:
                    f.write(image_bytes)
                print(f"강제 처리된 이미지 저장됨: {forced_filename}")
            except Exception as force_err:
                print(f"강제 처리 실패: {str(force_err)}")
        
        # 디버깅용으로 업로드 직전 이미지 저장
        try:
            debug_upload_filename = f"debug_upload_{camera_type}_{device_rotation}.jpg"
            with open(debug_upload_filename, "wb") as f:
                f.write(image_bytes)
            print(f"S3 업로드 직전 이미지 저장됨: {debug_upload_filename}")
        except Exception as save_err:
            print(f"업로드 직전 이미지 저장 실패: {str(save_err)}")
        
        # presigned URL을 통해 S3에 이미지 업로드
        upload_image_to_presigned_url(presigned_url, image_bytes)
        print(f"S3 업로드 완료: URL={photo_url}")

        upload_result = finalize_photo_upload(
            share_group_id=share_group_id,
            photo_url=photo_url,
            profile_id_list=profile_id_list,
            location=location,
            taked_at=take_at,
            image_bytes=image_bytes,  # 이미지 바이트가 정확히 전달되는지 확인
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