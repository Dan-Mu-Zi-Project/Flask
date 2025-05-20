import base64
from openai import OpenAI
import os
# from dotenv import load_dotenv # Flask 앱 설정에서 관리되므로 주석 처리 또는 삭제
import traceback # 오류 로깅을 위해 추가
from flask import Blueprint, request, jsonify, Response, stream_with_context
from flasgger import swag_from # swag_from 임포트 추가
from app.config import Config # Flask 앱 설정을 가져오기 위함
import cv2
import numpy as np
from PIL import Image
import io

# load_dotenv() # Flask 앱 초기화 시 한 번 로드하는 것이 일반적

img2text_bp = Blueprint("img2text", __name__, url_prefix="/test")

# OpenAI 클라이언트 초기화 (app.config 또는 Config 객체 사용)
# client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) # 기존 방식
client = OpenAI(api_key=Config.OPENAI_API_KEY) # 수정된 방식

# Function to encode the image (파일을 직접 받는 대신 파일 스트림을 받도록 수정)
def encode_image_stream(image_file_storage):
    return base64.b64encode(image_file_storage.read()).decode("utf-8")

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
        raise

@img2text_bp.route("/img2text", methods=["POST"])
@swag_from(os.path.join(os.path.dirname(__file__), "../../../docs/img2text.yml")) # Swagger YML 파일 경로 지정
def generate_text_from_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files["image"]
    # prompt_text = request.form.get("prompt", "이미지에 대해 짧게 설명해주세요.") # 기존 방식

    user_custom_prompt = request.form.get("prompt")
    if user_custom_prompt:
        prompt_text = user_custom_prompt.strip() + " 짧게 설명해주세요."
    else:
        prompt_text = "이미지에 대해 짧게 설명해주세요." # 사용자가 프롬프트를 제공하지 않은 경우의 기본값

    if not image_file.filename:
        return jsonify({"error": "No image file selected"}), 400
    
    # 카메라 타입과 기기 방향 정보 가져오기
    camera_type = request.form.get("camera_type", "back")  # 기본값은 후면 카메라
    device_rotation = request.form.get("device_rotation", "portraitUp")  # 기본값은 세로 정상
    
    try:
        # 이미지 파일 읽기
        image_bytes = image_file.read()
        image_file.seek(0)  # 파일 포인터 리셋
        
        # 처리가 필요한 경우 이미지 전처리
        if "no_preprocess" not in request.form:
            try:
                processed_image_bytes = preprocess_image(image_bytes, camera_type, device_rotation)
                
                # 전처리된 이미지가 유효한지 확인 (추가 유효성 검사)
                if processed_image_bytes and len(processed_image_bytes) > 0:
                    # 이미지 형식 확인을 위해 PIL로 다시 열어봄
                    try:
                        test_image = Image.open(io.BytesIO(processed_image_bytes))
                        print(f"검증: 전처리된 이미지 형식: {test_image.format}, 크기: {test_image.size}")
                        
                        # 전처리된 이미지를 base64로 인코딩
                        base64_image = base64.b64encode(processed_image_bytes).decode("utf-8")
                        print(f"Base64 인코딩된 이미지 문자열 길이: {len(base64_image)}")
                    except Exception as e:
                        print(f"전처리된 이미지 유효성 검사 실패: {str(e)}")
                        traceback.print_exc()
                        # 원본 이미지로 대체
                        image_file.seek(0)
                        base64_image = encode_image_stream(image_file)
                else:
                    print("전처리된 이미지가 비어있거나 유효하지 않음")
                    # 원본 이미지 사용
                    image_file.seek(0)
                    base64_image = encode_image_stream(image_file)
            except Exception as e:
                # 전처리 실패 시 원본 이미지 사용
                print(f"이미지 전처리 실패: {str(e)}")
                traceback.print_exc()
                image_file.seek(0)
                base64_image = encode_image_stream(image_file)
        else:
            # 전처리 무시 옵션이 있는 경우 원본 이미지 사용
            image_file.seek(0)
            base64_image = encode_image_stream(image_file)

        def stream_response_generator():
            try:
                # 이미지 MIME 타입을 명시적으로 지정
                image_mime = "image/jpeg"
                
                response = client.chat.completions.create(
                    model="gpt-4.1-nano-2025-04-14", # 모델은 기존 스크립트 설정 유지
                    stream=True,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text}, # 사용자 입력 프롬프트 사용
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{image_mime};base64,{base64_image}",
                                        "detail": "low"
                                    }
                                },
                            ],
                        }
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    n=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    seed=42,
                    max_tokens=100, # 필요에 따라 조절
                )

                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                traceback.print_exc()
                # 스트림 중간에 오류 발생 시 클라이언트에게 알릴 방법 고려 (예: 특정 에러 메시지 yield)
                yield f"\n이미지 분석 중 오류가 발생했습니다: {str(e)}\n"

        return Response(stream_with_context(stream_response_generator()), mimetype="text/plain")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to process image and generate text", "message": str(e)}), 500

# 객체 피드백 전용 엔드포인트 추가
@img2text_bp.route("/object-feedback", methods=["POST"])
def generate_object_feedback():
    try:
        data = request.json
        photo_id = data.get("photo_id")
        device_rotation = data.get("device_rotation", "portraitUp")
        
        # 실제 사용에서는 photo_id를 이용하여 저장된 이미지를 찾아야 함
        # 테스트용으로는 고정 응답 대신 실제 객체 인식을 모방한 응답 생성
        
        feedback_options = [
            "이미지에 보이는 물체는 책상 위의 노트북, 마우스, 그리고 커피 머그입니다. 노트북 화면에는 코드 편집기가 열려있습니다.",
            "이미지에는 거실이 보입니다. 소파와 TV, 그리고 커피 테이블이 있습니다. 창문을 통해 햇빛이 들어오고 있습니다.",
            "이미지에는 주방 공간이 보입니다. 싱크대와 전자레인지, 그리고 냉장고가 보입니다. 카운터 위에는 과일 바구니가 놓여있습니다.",
            "이미지에는 공원 풍경이 보입니다. 나무들과 잔디, 그리고 벤치가 있습니다. 원거리에 산책하는 사람들이 보입니다.",
            "이미지에는 책상 위에 여러 사무용품이 보입니다. 노트북, 키보드, 마우스, 그리고 몇 권의 책과 메모장이 있습니다."
        ]
        
        import random
        # 디바이스 회전 정보를 포함한 응답
        feedback = random.choice(feedback_options) + f" (기기 방향: {device_rotation})"
        
        print(f"Object feedback generated for photo_id: {photo_id}, rotation: {device_rotation}")
        
        return jsonify({"feedback": feedback, "success": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "객체 피드백 생성 실패", "message": str(e)}), 500

# 인물 피드백 전용 엔드포인트 추가
@img2text_bp.route("/people-feedback", methods=["POST"])
def generate_people_feedback():
    try:
        data = request.json
        photo_id = data.get("photo_id")
        device_rotation = data.get("device_rotation", "portraitUp")
        
        # 실제 사용에서는 photo_id를 이용하여 저장된 이미지를 찾아야 함
        # 테스트용으로는 고정 응답 대신 실제 인물 인식을 모방한 응답 생성
        
        feedback_options = [
            "이미지에는 두 명의 사람이 보입니다. 한 명은 안경을 쓰고 노트북을 보고 있으며, 다른 한 명은 서류를 들고 이야기하고 있습니다.",
            "이미지에는 세 명의 사람이 웃으며 서 있습니다. 모두 카메라를 향해 미소 짓고 있으며, 구도가 잘 잡혀 있습니다.",
            "이미지에는 한 명의 사람이 창가에 앉아 있습니다. 빛이 얼굴에 잘 비치고 있어 표정이 선명하게 보입니다.",
            "이미지에는 네 명의 사람들이 테이블을 둘러싸고 앉아 있습니다. 왼쪽 끝 사람은 살짝 프레임에서 잘려 있습니다. 모두 정면을 보고 있습니다.",
            "이미지에는 여러 명의 사람들이 단체 사진을 찍고 있습니다. 뒷줄 사람들은 뒤에서 점프하고 있으며, 앞줄은 앉아있습니다. 구도가 잘 잡혔습니다."
        ]
        
        import random
        # 디바이스 회전 정보를 포함한 응답
        feedback = random.choice(feedback_options) + f" (기기 방향: {device_rotation})"
        
        print(f"People feedback generated for photo_id: {photo_id}, rotation: {device_rotation}")
        
        return jsonify({"feedback": feedback, "success": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "인물 피드백 생성 실패", "message": str(e)}), 500

# 이미지 업로드 처리 엔드포인트 추가
@img2text_bp.route("/upload", methods=["POST"])
def upload_image():
    if "photo" not in request.files:
        return jsonify({"error": "이미지 파일이 제공되지 않았습니다"}), 400
    
    try:
        image_file = request.files["photo"]
        camera_type = request.form.get("camera_type", "back")
        device_rotation = request.form.get("device_rotation", "portraitUp")
        pitch = request.form.get("pitch", "0.0")
        roll = request.form.get("roll", "0.0")
        yaw = request.form.get("yaw", "0.0")
        
        # 회전 각도 계산
        rotation_angle = 0
        if device_rotation == "portraitUp":
            rotation_angle = 0
        elif device_rotation == "landscapeRight":
            rotation_angle = 90
        elif device_rotation == "portraitDown":
            rotation_angle = 180
        elif device_rotation == "landscapeLeft":
            rotation_angle = 270
        
        # 실제 프로덕션에서는 이미지를 저장하고 ID를 생성
        photo_id = f"img_{int(float(request.form.get('timestamp', '0')))}"
        
        # 테스트용 응답
        return jsonify({
            "success": True,
            "message": "이미지 업로드 성공",
            "photo_id": photo_id,
            "orientation": {
                "type": device_rotation,
                "description": request.form.get("orientation_desc", ""),
                "rotation": rotation_angle,
                "autoRotated": True
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "이미지 업로드 실패", "message": str(e)}), 500
