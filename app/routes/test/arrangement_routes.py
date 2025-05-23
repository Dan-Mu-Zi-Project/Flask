"""
YOLO 기반 단체 사진 배치 분석 라우트 모듈
"""
from flask import Blueprint, request, jsonify, current_app
from flasgger import swag_from
import numpy as np
import cv2
from PIL import Image
import io
import logging
from app.utils.people_arrangement import PeopleArrangementAdvisor
from app.utils.face_utils import extract_multiple_face_embeddings, find_best_match
from app.utils.upload_utils import request_group_face_vectors

arrangement_bp = Blueprint('arrangement', __name__, url_prefix='/test')

# PeopleArrangementAdvisor 인스턴스 생성
arrangement_advisor = PeopleArrangementAdvisor()

# 업로드 라우트에서 참고한 전처리용 ImageProcessor 클래스
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
            if camera_type == "back":
                if quarters != 0:
                    img_array = self.rotate_image_90(img_array, quarters)
            else:
                img_array = cv2.flip(img_array, 1)
                if quarters != 0:
                    img_array = self.rotate_image_90(img_array, quarters)
            result_image = Image.fromarray(img_array)
            if result_image.mode != 'RGB':
                result_image = result_image.convert('RGB')
            output_buffer = io.BytesIO()
            result_image.save(output_buffer, format="JPEG", quality=95)
            processed_bytes = output_buffer.getvalue()
            self.cache[cache_key] = processed_bytes
            return processed_bytes
        except Exception as e:
            current_app.logger.error(f"이미지 처리 오류: {str(e)}")
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
    def rotate_image_90(self, image, quarters):
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

image_processor = ImageProcessor()

# --- Feedback section toggles ---
ENABLE_DEPTH_FEEDBACK = True
ENABLE_REPOSITION_FEEDBACK = True
ENABLE_SPACING_FEEDBACK = False  # spacing은 항상 응답하지 않음

@arrangement_bp.route('/arrangement', methods=['POST'])
@swag_from({
    'tags': ['Image Analysis'],
    'summary': 'YOLO 기반 단체 사진 배치 분석 및 피드백',
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'image',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': '분석할 이미지 파일'
        },
        {
            'name': 'margin_ratio',
            'in': 'formData',
            'type': 'number',
            'required': False,
            'default': 0.10,
            'description': '좌우 여백 비율 (0.0 ~ 0.5)'
        },
        {
            'name': 'tol_ratio',
            'in': 'formData',
            'type': 'number',
            'required': False,
            'default': 0.20,
            'description': '이상적 간격 허용 오차 비율 (0.0 ~ 0.5)'
        },
        {
            'name': 'cameraType',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'default': 'back',
            'description': '카메라 타입 (back 또는 front)'
        },
        {
            'name': 'deviceRotation',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'default': 'portraitUp',
            'description': '기기 회전 방향'
        },
        {
            'name': 'shareGroupId',
            'in': 'formData',
            'type': 'integer',
            'required': False,
            'description': '그룹 ID'
        }
    ],
    'responses': {
        200: {
            'description': '배치 분석 결과',
            'schema': {
                'type': 'object',
                'properties': {
                    'people': {'type': 'array'},
                    'frame': {'type': 'object'},
                    'ideal_gap': {'type': 'number', 'nullable': True},
                    'feedback': {'type': 'object'},
                    'identified_people': {'type': 'array'},
                    'left_to_right_names': {'type': 'array'}
                }
            }
        },
        400: {'description': '잘못된 요청'},
        500: {'description': '서버 오류'}
    }
})
def analyze_arrangement():
    if 'image' not in request.files:
        return jsonify({'error': '이미지 파일이 제공되지 않았습니다'}), 400

    try:
        margin_ratio = float(request.form.get('margin_ratio', 0.10))
        if not 0 <= margin_ratio <= 0.5:
            return jsonify({'error': 'margin_ratio는 0.0 ~ 0.5 사이여야 합니다'}), 400
    except ValueError:
        return jsonify({'error': 'margin_ratio는 숫자여야 합니다'}), 400

    try:
        tol_ratio = float(request.form.get('tol_ratio', 0.20))
        if not 0 <= tol_ratio <= 0.5:
            return jsonify({'error': 'tol_ratio는 0.0 ~ 0.5 사이여야 합니다'}), 400
    except ValueError:
        return jsonify({'error': 'tol_ratio는 숫자여야 합니다'}), 400

    try:
        image_file = request.files['image']
        camera_type = request.form.get('cameraType', 'back')
        device_rotation = request.form.get('deviceRotation', 'portraitUp')
        original_image_bytes = image_file.read()
        processed_image_bytes = image_processor.process(
            original_image_bytes,
            camera_type,
            device_rotation
        )
        if processed_image_bytes:
            img_array = np.frombuffer(processed_image_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            img_array = np.frombuffer(original_image_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': '이미지를 읽을 수 없습니다'}), 400
        # --- 인물 식별 추가 ---
        face_results = extract_multiple_face_embeddings(processed_image_bytes or original_image_bytes)
        valid_embeddings = [r for r in face_results if "embedding" in r]
        identified_people = []
        if valid_embeddings:
            # 그룹 벡터 조회 (shareGroupId, access_token은 필요시 form/header에서 받아야 함)
            share_group_id = request.form.get("shareGroupId", type=int)
            access_token = request.headers.get("Authorization")
            if access_token:
                access_token = access_token.replace("Bearer ", "").strip()
            if share_group_id and access_token:
                group_vector_response = request_group_face_vectors(share_group_id, access_token)
                for face in valid_embeddings:
                    matches = find_best_match(face["embedding"], group_vector_response)
                    # 임베딩 정보 제외
                    face_info = {k: v for k, v in face.items() if k != "embedding"}
                    if matches:
                        identified_people.append({"face": face_info, "match": matches[0]})
                    else:
                        identified_people.append({"face": face_info, "match": None})
            else:
                # 임베딩 정보 제외
                identified_people = [{"face": {k: v for k, v in face.items() if k != "embedding"}, "match": None} for face in valid_embeddings]
        # --- 배치 분석 ---
        result = arrangement_advisor.analyze(img, margin_ratio=margin_ratio, tol_ratio=tol_ratio)
        result["identified_people"] = identified_people

        # --- people의 id/target을 이름으로 통합 ---
        if "people" in result and identified_people:
            def get_center(p):
                if "center" in p and isinstance(p["center"], (list, tuple)):
                    return p["center"]
                elif "bbox" in p and isinstance(p["bbox"], (list, tuple)):
                    x, y, w, h = p["bbox"]
                    return [x + w/2, y + h/2]
                elif "pos" in p and isinstance(p["pos"], dict):
                    return [p["pos"].get("x", 0), p["pos"].get("y", 0)]
                return [0,0]
            def get_face_center(face):
                box = face["face"].get("box") or face["face"].get("bbox")
                if box and isinstance(box, (list,tuple)):
                    x, y, w, h = box
                    return [x + w/2, y + h/2]
                return [0,0]
            used = set()
            name_map = {}
            for p in result["people"]:
                pc = get_center(p)
                min_idx = -1
                min_dist = float("inf")
                for idx, face in enumerate(identified_people):
                    if idx in used:
                        continue
                    fc = get_face_center(face)
                    dist = (pc[0]-fc[0])**2 + (pc[1]-fc[1])**2
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = idx
                if min_idx >= 0:
                    used.add(min_idx)
                    match = identified_people[min_idx].get("match")
                    name = match.get("name") if match and match.get("name") else match.get("profileId") if match else None
                    # people의 id/target을 이름으로 통합
                    p["name"] = name
                    name_map[p.get("id")] = name
                    if "target" in p:
                        del p["target"]
                    if "id" in p:
                        del p["id"]
                else:
                    p["name"] = None
            # --- feedback 내 id/target도 이름으로 변환 ---
            fb = result.get("feedback", {})
            # reposition
            for r in fb.get("reposition", []):
                if "target" in r:
                    import re
                    m = re.match(r"(\d+)", str(r["target"]))
                    if m:
                        pid = int(m.group(1))
                        r["target"] = name_map.get(pid, r["target"])
            # spacing
            for s in fb.get("spacing", []):
                if "between" in s:
                    import re
                    m = re.match(r"(\d+)[^\d]+(\d+)", str(s["between"]))
                    if m:
                        pid1 = int(m.group(1))
                        pid2 = int(m.group(2))
                        n1 = name_map.get(pid1, s["between"])
                        n2 = name_map.get(pid2, s["between"])
                        s["between"] = f"{n1}-{n2}"
            # depth
            for d in fb.get("depth", []):
                if "id" in d:
                    n = name_map.get(d["id"], d["id"])
                    d["name"] = n
                    del d["id"]
            # --- reposition 피드백 그룹화 및 자연어 문장 생성 ---
            reposition_grouped = []
            if "reposition" in fb:
                from collections import defaultdict
                group_map = defaultdict(list)
                for r in fb["reposition"]:
                    key = (r.get("direction"), r.get("instruction"))
                    group_map[key].append(r.get("target"))
                for (direction, instruction), targets in group_map.items():
                    # 이름이 None인 경우 '이름없음'으로 대체, 이름이 있으면 성 빼기
                    def get_first_name(name):
                        if name and isinstance(name, str) and len(name) > 1:
                            return name[1:] if len(name) > 2 else name[-1]
                        return name if name else '이름없음'
                    name_str = ', '.join([get_first_name(t) for t in targets])
                    # '은/는' 조사 처리
                    if name_str.endswith('은') or name_str.endswith('는'):
                        name_str2 = name_str
                    else:
                        name_str2 = name_str + '은'
                    sentence = f"{name_str2} {instruction}"
                    reposition_grouped.append({
                        "targets": targets,
                        "direction": direction,
                        "instruction": instruction,
                        "message": sentence
                    })
                fb["reposition_grouped"] = reposition_grouped
                del fb["reposition"]
            # --- depth 피드백 그룹화 및 자연어 문장 생성 ---
            depth_grouped = []
            if "depth" in fb:
                from collections import defaultdict
                group_map = defaultdict(list)
                for d in fb["depth"]:
                    key = d.get("instruction")
                    group_map[key].append(d.get("name"))
                for instruction, names in group_map.items():
                    # 이동지시만 추출 (예: "카메라에 너무 가깝습니다. 뒤로 조금 이동해주세요.")
                    move_part = None
                    for move_kw in ["앞으로 조금 이동해주세요", "뒤로 조금 이동해주세요"]:
                        if move_kw in instruction:
                            move_part = move_kw
                            break
                    if not move_part:
                        continue  # 이동지시가 없으면 메시지 생성하지 않음
                    def get_first_name(name):
                        if name and isinstance(name, str) and len(name) > 1:
                            return name[1:] if len(name) > 2 else name[-1]
                        return name if name else '이름없음'
                    name_str = ', '.join([get_first_name(n) for n in names])
                    if name_str.endswith('은') or name_str.endswith('는'):
                        name_str2 = name_str
                    else:
                        name_str2 = name_str + '은'
                    sentence = f"{name_str2} {move_part}"
                    depth_grouped.append({
                        "targets": names,
                        "instruction": move_part,
                        "message": sentence
                    })
                fb["depth_grouped"] = depth_grouped
                del fb["depth"]
            # --- feedback section toggles ---
            if not ENABLE_REPOSITION_FEEDBACK and "reposition_grouped" in fb:
                del fb["reposition_grouped"]
            if not ENABLE_DEPTH_FEEDBACK and "depth_grouped" in fb:
                del fb["depth_grouped"]
            if not ENABLE_SPACING_FEEDBACK and "spacing" in fb:
                del fb["spacing"]
            result["feedback"] = fb
        # --- 왼쪽부터 인물 이름 정렬 ---
        left_to_right_names = []
        if "people" in result and identified_people:
            people_sorted = sorted(result["people"], key=lambda p: get_center(p)[0])
            used = set()
            for p in people_sorted:
                pc = get_center(p)
                min_idx = -1
                min_dist = float("inf")
                for idx, face in enumerate(identified_people):
                    if idx in used:
                        continue
                    fc = get_face_center(face)
                    dist = (pc[0]-fc[0])**2 + (pc[1]-fc[1])**2
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = idx
                if min_idx >= 0:
                    used.add(min_idx)
                    match = identified_people[min_idx].get("match")
                    name = match.get("name") if match and match.get("name") else match.get("profileId") if match else None
                    left_to_right_names.append(name)
                else:
                    left_to_right_names.append(None)
        result["left_to_right_names"] = left_to_right_names
        # --- 응답에서 ideal_gap, identified_people, people 제외 ---
        for k in ["ideal_gap", "identified_people", "people"]:
            if k in result:
                del result[k]
        return jsonify(result)
    except Exception as e:
        current_app.logger.error(f'배치 분석 실패: {str(e)}')
        return jsonify({'error': f'배치 분석 실패: {str(e)}'}), 500

@arrangement_bp.route('/arrangement/only', methods=['POST'])
@swag_from({
    'tags': ['Image Analysis'],
    'summary': 'YOLO 기반 단체 사진 배치 분석만 수행 (피드백 없이 결과만 반환)',
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'image',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': '분석할 이미지 파일'
        },
        {
            'name': 'margin_ratio',
            'in': 'formData',
            'type': 'number',
            'required': False,
            'default': 0.10,
            'description': '좌우 여백 비율 (0.0 ~ 0.5)'
        },
        {
            'name': 'tol_ratio',
            'in': 'formData',
            'type': 'number',
            'required': False,
            'default': 0.20,
            'description': '이상적 간격 허용 오차 비율 (0.0 ~ 0.5)'
        },
        {
            'name': 'cameraType',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'default': 'back',
            'description': '카메라 타입 (back 또는 front)'
        },
        {
            'name': 'deviceRotation',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'default': 'portraitUp',
            'description': '기기 회전 방향'
        }
    ],
    'responses': {
        200: {
            'description': '배치 분석 결과만 반환',
            'schema': {
                'type': 'object',
                'properties': {
                    'people': {'type': 'array'},
                    'frame': {'type': 'object'},
                    'ideal_gap': {'type': 'number', 'nullable': True}
                }
            }
        },
        400: {'description': '잘못된 요청'},
        500: {'description': '서버 오류'}
    }
})
def arrangement_only():
    if 'image' not in request.files:
        return jsonify({'error': '이미지 파일이 제공되지 않았습니다'}), 400

    try:
        margin_ratio = float(request.form.get('margin_ratio', 0.10))
        if not 0 <= margin_ratio <= 0.5:
            return jsonify({'error': 'margin_ratio는 0.0 ~ 0.5 사이여야 합니다'}), 400
    except ValueError:
        return jsonify({'error': 'margin_ratio는 숫자여야 합니다'}), 400

    try:
        tol_ratio = float(request.form.get('tol_ratio', 0.20))
        if not 0 <= tol_ratio <= 0.5:
            return jsonify({'error': 'tol_ratio는 0.0 ~ 0.5 사이여야 합니다'}), 400
    except ValueError:
        return jsonify({'error': 'tol_ratio는 숫자여야 합니다'}), 400

    try:
        image_file = request.files['image']
        camera_type = request.form.get('cameraType', 'back')
        device_rotation = request.form.get('deviceRotation', 'portraitUp')
        original_image_bytes = image_file.read()
        processed_image_bytes = image_processor.process(
            original_image_bytes,
            camera_type,
            device_rotation
        )
        if processed_image_bytes:
            img_array = np.frombuffer(processed_image_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            img_array = np.frombuffer(original_image_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': '이미지를 읽을 수 없습니다'}), 400

        result = arrangement_advisor.analyze(img, margin_ratio=margin_ratio, tol_ratio=tol_ratio)
        result_simple = {k: v for k, v in result.items() if k in ['people', 'frame', 'ideal_gap']}
        return jsonify(result_simple)

    except Exception as e:
        current_app.logger.error(f'배치 분석 실패: {str(e)}')
        return jsonify({'error': f'배치 분석 실패: {str(e)}'}), 500
    



