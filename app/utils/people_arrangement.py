"""
YOLO-기반 단체 사진 배치 분석 및 피드백 유틸리티
-------------------------------------------------
• ultralytics YOLOv8 pose 모델을 이용해 사람들을 탐지(코 위치 사용)
• 프레임 전체를 고려하여 ‑ 좌우 여백 10 % 를 유지한 뒤, 나머지 80 % 구간을 균등 분할해 이상적 x 좌표(target_x) 결정
• 인물 간 실제 간격이 이상적 간격보다 너무 좁거나 넓으면 추가 간격 피드백 제공
• 완전히 독립적인 모듈이므로 다른 모듈에 의존하지 않음.

사용 예시:
>>> import cv2, json
>>> from app.utils.people_arrangement import PeopleArrangementAdvisor
>>> advisor = PeopleArrangementAdvisor()
>>> img = cv2.imread('group.jpg')
>>> result = advisor.analyze(img)
>>> print(json.dumps(result["feedback"], ensure_ascii=False, indent=2))
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("ultralytics 패키지가 설치되어 있지 않습니다. `pip install ultralytics` 후 사용하세요.") from e


class PeopleArrangementAdvisor:
    """YOLO-pose 를 활용한 단체 사진 배치 분석 및 개선 어드바이저"""

    # ---------------------- 초기화 및 모델 로드 ---------------------- #
    def __init__(self,
                 model_name: str = "yolov8n-pose",
                 model_dir: str | Path | None = None,
                 arrangement_mode: str = "group"):  # 'group', 'visual', 'dynamic', 'context'
        """YOLO 모델을 로드합니다.

        Args:
            model_name: ultralytics Model 이름 (default: yolov8n-pose)
            model_dir: 학습된 모델을 저장할 디렉터리. None 이면 현재 파일 기준 ./models
            arrangement_mode: 위치 계산 방식 ('group', 'visual', 'dynamic', 'context')
        """
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / f"{model_name}.pt"
        self.arrangement_mode = arrangement_mode

        if model_path.exists():
            self.model = YOLO(str(model_path))
        else:
            # 최초 실행 시 인터넷에서 모델 다운로드 후 저장
            self.model = YOLO(model_name)
            self.model.save(str(model_path))

    # ---------------------- Public API ---------------------- #
    def analyze(self, image: np.ndarray, *, margin_ratio: float = 0.10, tol_ratio: float = 0.2) -> Dict[str, Any]:
        """이미지를 분석해 인물 위치·간격·배치 피드백을 반환합니다.

        Args:
            image: BGR (cv2.imread) 형태의 이미지 배열
            margin_ratio: 좌우 여백 비율 (default 10 %)
            tol_ratio: 이상적 간격 허용 오차 비율 (default ±20 %)

        Returns:
            Dict 구조
            {
              "people": [ {"id": 1, "pos": {"x":.., "y":..}, "bbox": (x1,y1,x2,y2)} , ... ],
              "frame": {"w": int, "h": int},
              "ideal_gap": float,
              "feedback": {
                   "reposition": [ {...} ],   # 각 인물 이동 제안
                   "spacing": [ {...} ],       # 간격 관련 추가 제안
                   "depth": [ {...} ]          # 깊이 관련 추가 제안
              }
            }
        """
        if image is None:
            raise ValueError("image 매개변수가 None 입니다. cv2.imread 결과를 전달하세요.")

        h, w = image.shape[:2]
        frame_center_x = w / 2

        # ---------------- YOLO 추론 ----------------
        yolo_res = self.model(image, conf=0.5, verbose=False)[0]

        # 사람 클래스 id == 0
        people_boxes, nose_points = self._extract_people_boxes_and_noses(yolo_res)
        people_info: List[Dict[str, Any]] = []
        for idx, (bbox, nose) in enumerate(zip(people_boxes, nose_points), start=1):
            people_info.append({
                "id": idx,
                "pos": {"x": float(nose[0]), "y": float(nose[1])},
                "bbox": tuple(map(float, bbox))
            })

        total_people = len(people_info)
        if total_people == 0:
            return {
                "people": [],
                "frame": {"w": w, "h": h},
                "ideal_gap": None,
                "feedback": {
                    "reposition": [],
                    "spacing": [],
                    "depth": []
                }
            }

        # 위치 계산 방식 선택
        if total_people == 1:
            target_xs = [frame_center_x]
            ideal_gap = None
        else:
            margin = margin_ratio * w
            usable_width = w - 2 * margin
            
            if self.arrangement_mode == "group":
                target_xs = self._calculate_group_based_positions(people_info, usable_width)
            elif self.arrangement_mode == "visual":
                target_xs = self._calculate_visual_balance_positions(people_info, usable_width)
            elif self.arrangement_mode == "dynamic":
                target_xs = self._calculate_dynamic_spacing(people_info, usable_width)
            elif self.arrangement_mode == "context":
                target_xs = self._calculate_context_based_positions(people_info, usable_width)
            else:
                # 기본 균등 분할
                target_xs = [margin + (i + 0.5) * (usable_width / total_people) 
                           for i in range(total_people)]
            
            target_xs = [x + margin for x in target_xs]  # 여백 적용
            ideal_gap = np.mean([target_xs[i+1] - target_xs[i] for i in range(len(target_xs)-1)])

        # ---------------- 1) 개별 인물 이동 피드백 ----------------
        reposition_fb = []
        people_sorted = sorted(people_info, key=lambda p: p["pos"]["x"])
        target_xs_sorted = sorted(target_xs)  # target_xs도 정렬
        
        for person, tx in zip(people_sorted, target_xs_sorted):
            dx = tx - person["pos"]["x"]
            if abs(dx) < 20:  # 20px 미만은 무시
                continue
                
            # 이동 방향 결정 - dx의 부호를 반대로 적용
            dir_str = "왼쪽" if dx > 0 else "오른쪽"  # 방향을 반대로 변경
            reposition_fb.append({
                "target": f"{person['id']}번",
                "instruction": f"{dir_str}으로 조금 이동해주세요.",
                "direction": dir_str,
                "dx": round(dx, 1)
            })

        # ---------------- 2) 인물 간 간격 피드백 ----------------
        spacing_fb = []
        if total_people >= 2 and ideal_gap is not None:
            # 간격이 너무 좁을 때는 기준을 더 완화 (기존의 0.5배)
            tol_min = (1 - tol_ratio * 0.5) * ideal_gap
            # 간격이 너무 멀 때는 기존 기준 유지
            tol_max = (1 + tol_ratio) * ideal_gap
            
            # 인물들을 x 기준 정렬
            people_sorted = sorted(people_info, key=lambda p: p["pos"]["x"])
            xs = [p["pos"]["x"] for p in people_sorted]
            ids = [p["id"] for p in people_sorted]
            
            # 바로 옆 사람과의 간격만 체크
            for i in range(total_people - 1):
                gap = xs[i+1] - xs[i]
                if gap < tol_min:
                    spacing_fb.append({
                        "between": f"{ids[i]}번-{ids[i+1]}번",
                        "issue": "간격 좁음",
                        "action": "멀리",
                        "instruction": "두 사람 사이가 너무 좁습니다. 서로 멀리 이동해주세요.",
                        "gap": None if np.isnan(gap) else round(gap, 1)
                    })
                elif gap > tol_max:
                    spacing_fb.append({
                        "between": f"{ids[i]}번-{ids[i+1]}번",
                        "issue": "간격 넓음",
                        "action": "가까이",
                        "instruction": "두 사람 사이가 너무 멀습니다. 서로 가까이 이동해주세요.",
                        "gap": None if np.isnan(gap) else round(gap, 1)
                    })

        # ---------------- 3) 깊이 피드백 ----------------
        depth_fb = []
        # 바운딩박스 높이 기반 상대 깊이 평가
        heights = [p["bbox"][3] - p["bbox"][1] for p in people_info]
        if heights:
            median_h = np.median(heights)
            tol_depth = 0.15  # 15% 허용 오차
            for person, h in zip(people_info, heights):
                if h > (1 + tol_depth) * median_h:
                    relative = (h/median_h) - 1
                    depth_fb.append({
                        "id": person["id"],
                        "issue": "깊이 가까움",
                        "action": "뒤로",
                        "instruction": "카메라에 너무 가깝습니다. 뒤로 조금 이동해주세요.",
                        "relative": None if np.isnan(relative) else round(relative, 2)
                    })
                elif h < (1 - tol_depth) * median_h:
                    relative = 1 - (h/median_h)
                    depth_fb.append({
                        "id": person["id"],
                        "issue": "깊이 멀음",
                        "action": "앞으로",
                        "instruction": "카메라에서 너무 멀리 있습니다. 앞으로 조금 이동해주세요.",
                        "relative": None if np.isnan(relative) else round(relative, 2)
                    })

        result = {
            "people": people_info,
            "frame": {"w": w, "h": h},
            "ideal_gap": None if ideal_gap is None or np.isnan(ideal_gap) else round(ideal_gap, 1),
            "feedback": {
                "reposition": reposition_fb,
                "spacing": spacing_fb,
                "depth": depth_fb
            }
        }

        # JSON 직렬화 시 한글 처리
        return json.loads(json.dumps(result, ensure_ascii=False))

    # ---------------------- Helper Methods ---------------------- #
    @staticmethod
    def _extract_people_boxes_and_noses(yolo_res) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float]]]:
        """YOLO 결과에서 사람 바운딩박스와 코 위치(keypoint #0)를 추출"""
        boxes, noses = [], []
        for i, det in enumerate(yolo_res.boxes.data):
            if det[5] != 0:  # class 0 == person
                continue
            x1, y1, x2, y2 = map(float, det[:4].tolist())
            kp = yolo_res.keypoints.data[i]
            nose = kp[0].tolist()  # (x, y, conf)
            if float(nose[2]) < 0.2:  # nose confidence 너무 낮으면 중심으로 bbox 중간 사용
                nose_x = (x1 + x2) / 2
                nose_y = (y1 + y2) / 2
                nose = (nose_x, nose_y, 1.0)
            boxes.append((x1, y1, x2, y2))
            noses.append((nose[0], nose[1]))
        return boxes, noses

    def _calculate_group_based_positions(self, people_info: List[Dict[str, Any]], frame_width: int) -> List[float]:
        """그룹을 고려한 위치 계산"""
        # 1. 키포인트 기반으로 그룹 식별
        groups = self._identify_groups(people_info)
        
        # 2. 각 그룹 내에서의 상대적 위치 계산
        group_positions = {}
        for group_id, members in groups.items():
            # 그룹 내에서의 상대적 위치 계산
            relative_positions = self._calculate_relative_positions(members)
            group_positions[group_id] = relative_positions
        
        # 3. 그룹 간 간격 조정
        return self._adjust_group_spacing(group_positions, frame_width)

    def _identify_groups(self, people_info: List[Dict[str, Any]]) -> Dict[int, List[Dict]]:
        """키포인트 기반으로 그룹 식별"""
        groups = {}
        current_group = []
        
        for person in people_info:
            if not current_group:
                current_group.append(person)
                continue
                
            # 이전 사람과의 관계 분석
            last_person = current_group[-1]
            if self._are_related(last_person, person):
                current_group.append(person)
            else:
                groups[len(groups)] = current_group
                current_group = [person]
        
        if current_group:
            groups[len(groups)] = current_group
            
        return groups

    def _are_related(self, person1: Dict[str, Any], person2: Dict[str, Any]) -> bool:
        """두 사람이 관련이 있는지 확인"""
        # 키포인트 간의 거리 계산
        kp1 = self._extract_important_keypoints(person1)
        kp2 = self._extract_important_keypoints(person2)
        
        if not kp1 or not kp2:
            return False
            
        # 중심점 간의 거리 계산
        center1 = self._calculate_body_center(kp1)
        center2 = self._calculate_body_center(kp2)
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # 거리가 임계값 이하면 관련 있다고 판단
        return distance < 200  # 임계값은 조정 가능

    def _calculate_relative_positions(self, members: List[Dict[str, Any]]) -> List[float]:
        """그룹 내 상대적 위치 계산"""
        if not members:
            return []
            
        # 현재 x 좌표 기준으로 정렬
        sorted_members = sorted(members, key=lambda p: p["pos"]["x"])
        
        # 상대적 위치 계산
        positions = []
        for i, member in enumerate(sorted_members):
            relative_pos = i / (len(members) - 1) if len(members) > 1 else 0.5
            positions.append(relative_pos)
            
        return positions

    def _adjust_group_spacing(self, group_positions: Dict[int, List[float]], frame_width: int) -> List[float]:
        """그룹 간 간격 조정"""
        all_positions = []
        current_pos = 0
        
        for group_id, positions in group_positions.items():
            # 그룹 내 위치를 실제 픽셀 위치로 변환
            group_width = frame_width / len(group_positions)
            for pos in positions:
                all_positions.append(current_pos + pos * group_width)
            current_pos += group_width
            
        return all_positions

    def _calculate_visual_balance_positions(self, people_info: List[Dict[str, Any]], frame_width: int) -> List[float]:
        """시각적 균형을 고려한 위치 계산"""
        # 1. 각 사람의 시각적 무게 계산
        visual_weights = []
        for person in people_info:
            weight = self._calculate_visual_weight(person)
            visual_weights.append(weight)
        
        # 2. 시각적 중심점 계산
        center_of_gravity = self._calculate_center_of_gravity(people_info, visual_weights)
        
        # 3. 균형을 맞추는 위치 계산
        return self._balance_positions(people_info, visual_weights, center_of_gravity, frame_width)

    def _calculate_visual_weight(self, person: Dict[str, Any]) -> float:
        """시각적 무게 계산"""
        # 바운딩 박스 크기
        bbox = person["bbox"]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # 키포인트 신뢰도
        keypoints = person.get("keypoints", [])
        confidence = np.mean([kp[2] for kp in keypoints]) if keypoints else 0.5
        
        return area * confidence

    def _calculate_center_of_gravity(self, people_info: List[Dict[str, Any]], weights: List[float]) -> float:
        """시각적 중심점 계산"""
        total_weight = sum(weights)
        if total_weight == 0:
            return frame_width / 2
            
        weighted_sum = sum(p["pos"]["x"] * w for p, w in zip(people_info, weights))
        return weighted_sum / total_weight

    def _balance_positions(self, people_info: List[Dict[str, Any]], weights: List[float], 
                         center_of_gravity: float, frame_width: int) -> List[float]:
        """균형을 맞추는 위치 계산"""
        positions = []
        total_weight = sum(weights)
        
        # 각 사람의 위치를 중심점 기준으로 조정
        for person, weight in zip(people_info, weights):
            # 현재 위치와 중심점의 차이
            offset = person["pos"]["x"] - center_of_gravity
            
            # 무게에 비례하여 조정
            adjustment = offset * (weight / total_weight)
            
            # 새로운 위치 계산
            new_pos = person["pos"]["x"] - adjustment
            positions.append(new_pos)
            
        return positions

    def _calculate_dynamic_spacing(self, people_info: List[Dict[str, Any]], frame_width: int) -> List[float]:
        """역동적인 간격 조정"""
        # 1. 기본 간격 계산
        base_spacing = frame_width / (len(people_info) + 1)
        
        # 2. 각 사람의 특성에 따른 간격 조정
        adjusted_spacings = []
        for i, person in enumerate(people_info):
            # 키포인트 기반 특성 분석
            characteristics = self._analyze_characteristics(person)
            
            # 특성에 따른 간격 조정
            spacing_factor = self._calculate_spacing_factor(characteristics)
            adjusted_spacing = base_spacing * spacing_factor
            adjusted_spacings.append(adjusted_spacing)
        
        # 3. 위치 계산 - 전체 프레임을 고려한 균등 분포
        positions = []
        total_spacing = sum(adjusted_spacings)
        margin = (frame_width - total_spacing) / 2  # 양쪽 여백
        
        # 프레임 경계를 고려한 위치 계산
        current_pos = margin
        for spacing in adjusted_spacings:
            # 프레임 경계를 벗어나지 않도록 조정
            if current_pos < margin:
                current_pos = margin
            elif current_pos + spacing > frame_width - margin:
                current_pos = frame_width - margin - spacing
                
            positions.append(current_pos)
            current_pos += spacing
        
        return positions

    def _analyze_characteristics(self, person: Dict[str, Any]) -> Dict[str, float]:
        """키포인트 기반 특성 분석"""
        characteristics = {
            "height": 0.0,
            "confidence": 0.0,
            "pose_complexity": 0.0
        }
        
        # 키포인트 추출
        keypoints = self._extract_important_keypoints(person)
        if not keypoints:
            return characteristics
            
        # 높이 계산
        bbox = person["bbox"]
        characteristics["height"] = bbox[3] - bbox[1]
        
        # 신뢰도 계산
        characteristics["confidence"] = np.mean([kp[2] for kp in keypoints])
        
        # 포즈 복잡도 계산 (키포인트 간의 거리 분산)
        distances = []
        for i in range(len(keypoints)):
            for j in range(i+1, len(keypoints)):
                dist = np.sqrt((keypoints[i][0] - keypoints[j][0])**2 + 
                             (keypoints[i][1] - keypoints[j][1])**2)
                distances.append(dist)
        characteristics["pose_complexity"] = np.var(distances) if distances else 0
        
        return characteristics

    def _calculate_spacing_factor(self, characteristics: Dict[str, float]) -> float:
        """특성에 따른 간격 조정 계수 계산"""
        # 각 특성의 가중치
        weights = {
            "height": 0.4,
            "confidence": 0.3,
            "pose_complexity": 0.3
        }
        
        # 정규화된 특성 값 계산
        normalized = {
            "height": min(characteristics["height"] / 500, 1.0),  # 최대 500px 기준
            "confidence": characteristics["confidence"],
            "pose_complexity": min(characteristics["pose_complexity"] / 1000, 1.0)  # 최대 1000 기준
        }
        
        # 가중 평균 계산
        factor = sum(normalized[k] * weights[k] for k in weights)
        
        # 0.8 ~ 1.2 범위로 조정
        return 0.8 + (factor * 0.4)

    def _calculate_context_based_positions(self, people_info: List[Dict[str, Any]], frame_width: int) -> List[float]:
        """컨텍스트를 고려한 위치 계산"""
        # 1. 얼굴 방향 분석
        face_directions = self._analyze_face_directions(people_info)
        
        # 2. 키포인트 기반 관계 분석
        relationships = self._analyze_relationships(people_info)
        
        # 3. 컨텍스트 기반 위치 계산
        positions = []
        for i, person in enumerate(people_info):
            # 얼굴 방향과 관계를 고려한 위치 계산
            context_factor = self._calculate_context_factor(
                face_directions[i],
                relationships[i]
            )
            
            # 기본 위치에 컨텍스트 팩터 적용
            base_pos = (i + 0.5) * (frame_width / len(people_info))
            adjusted_pos = base_pos * context_factor
            positions.append(adjusted_pos)
        
        return positions

    def _analyze_face_directions(self, people_info: List[Dict[str, Any]]) -> List[float]:
        """얼굴 방향 분석"""
        directions = []
        for person in people_info:
            # 키포인트에서 얼굴 관련 포인트 추출
            keypoints = self._extract_important_keypoints(person)
            if not keypoints:
                directions.append(0.0)
                continue
                
            # 얼굴 방향 계산 (예: 어깨와 코의 상대적 위치)
            # 실제 구현에서는 더 정교한 얼굴 방향 감지 필요
            direction = 0.0  # 기본값
            directions.append(direction)
            
        return directions

    def _analyze_relationships(self, people_info: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """키포인트 기반 관계 분석"""
        relationships = []
        for person in people_info:
            # 다른 사람들과의 관계 분석
            relationship = {
                "left": 0.0,
                "right": 0.0,
                "center": 0.0
            }
            
            # 실제 구현에서는 더 정교한 관계 분석 필요
            relationships.append(relationship)
            
        return relationships

    def _calculate_context_factor(self, face_direction: float, relationship: Dict[str, float]) -> float:
        """컨텍스트 팩터 계산"""
        # 얼굴 방향과 관계를 고려한 팩터 계산
        # 실제 구현에서는 더 정교한 계산 필요
        return 1.0  # 기본값

    def _extract_important_keypoints(self, person: Dict[str, Any]) -> List[Tuple[float, float]]:
        """중요 키포인트 추출 (어깨, 엉덩이 등)"""
        # YOLO pose 모델의 키포인트에서 중요 포인트 추출
        keypoints = person.get("keypoints", [])
        important_points = []
        for kp in keypoints:
            if kp[2] > 0.5:  # 신뢰도가 높은 키포인트만 사용
                important_points.append((kp[0], kp[1]))
        return important_points

    def _calculate_body_center(self, keypoints: List[Tuple[float, float]]) -> Tuple[float, float]:
        """키포인트를 기반으로 신체 중심점 계산"""
        if not keypoints:
            return (0, 0)
        x_coords = [kp[0] for kp in keypoints]
        y_coords = [kp[1] for kp in keypoints]
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

    def _calculate_distribution_weights(self, center_points: List[Tuple[float, float]]) -> List[float]:
        """중심점 분포를 기반으로 가중치 계산"""
        if not center_points:
            return [1.0]
        
        # 중심점 간의 거리를 기반으로 가중치 계산
        weights = []
        for i, point in enumerate(center_points):
            # 주변 점들과의 거리 계산
            distances = []
            for j, other_point in enumerate(center_points):
                if i != j:
                    dist = np.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2)
                    distances.append(dist)
            
            # 거리가 멀수록 가중치 증가 (균등 분포를 위해)
            if distances:
                weight = np.mean(distances)  # 거리가 멀수록 가중치 증가
            else:
                weight = 1.0
            weights.append(weight)
        
        # 가중치 정규화
        max_weight = max(weights)
        if max_weight > 0:
            weights = [w / max_weight for w in weights]
        
        return weights


# -------------------- 테스트 실행 -------------------- #
if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(description="단체 사진 배치 분석 유틸리티")
    parser.add_argument("image", help="분석할 이미지 경로")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print("이미지를 불러올 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    advisor = PeopleArrangementAdvisor(arrangement_mode="group")
    result = advisor.analyze(img)
    print(json.dumps(result, ensure_ascii=False, indent=2))




    