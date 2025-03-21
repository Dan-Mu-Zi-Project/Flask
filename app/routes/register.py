from flask import Blueprint, request, jsonify
from flasgger import swag_from
import numpy as np
import os
import cv2
import json
import requests
from app.utils.api_helpers import build_external_url
from deepface import DeepFace

register_buffer = {}

FACE_CONFIDENCE_THRESHOLD = 0.82
REQUIRED_ANGLES = {"FRONT", "LEFT", "RIGHT"}
END_POINT = "/members/sampleImage"

model1_bp = Blueprint('model1_bp', __name__)

@model1_bp.route('/register', methods=['POST'])
@swag_from(os.path.join(os.path.dirname(__file__), '../../docs/register.yml'))
def register():
    EXTERNAL_API_URL = build_external_url(END_POINT)

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400

    file = request.files["image"]
    angle_type = request.form.get("angle_type", "").strip().upper()

    if angle_type not in REQUIRED_ANGLES:
        return jsonify({"success": False, "error": "Invalid or missing angle_type"}), 400

    auth_header = request.headers.get("Authorization")

    if not auth_header:
        return jsonify({"success": False, "error": "Missing or invalid access token"}), 401

    access_token = auth_header.replace("Bearer ", "").strip()
    user_token_id = access_token  # 메모리 기반 버퍼의 키로 사용

    if not file.mimetype.startswith("image/"):
        return jsonify({"success": False, "error": "Invalid file type."}), 400

    try:
        image_bytes = file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "error": "Invalid image data."}), 400

        result = DeepFace.represent(
            img_path=img,
            model_name="Facenet",
            detector_backend="yolov11n"
        )

        face_confidence = result[0].get("face_confidence", 0.0)
        if face_confidence < FACE_CONFIDENCE_THRESHOLD:
            return jsonify({
                "success": False,
                "error": f"Face confidence too low: {face_confidence:.4f}"
            }), 400

        embedding = result[0]["embedding"]
        face_vector = json.dumps(embedding)

        if user_token_id not in register_buffer:
            register_buffer[user_token_id] = {}

        register_buffer[user_token_id][angle_type] = {
            "angleType": angle_type,
            "faceVector": face_vector
        }

        response_payload = {
            "success": True,
            "angle_type": angle_type,
            "registered_angles": list(register_buffer[user_token_id].keys()),
            "all_angles_completed": False
        }

        if REQUIRED_ANGLES.issubset(register_buffer[user_token_id].keys()):
            try:
                face_sample_list = list(register_buffer[user_token_id].values())

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {access_token}"
                }

                payload = {
                    "faceSampleList": face_sample_list
                }
                print(EXTERNAL_API_URL)
                response = requests.post(
                    EXTERNAL_API_URL,
                    data=json.dumps(payload),
                    headers=headers
                )

                if response.status_code == 200:
                    response_payload["all_angles_completed"] = True
                    response_payload["external_sync"] = "success"
                else:
                    response_payload["external_sync"] = f"failed: {response.status_code} - {response.text}"

            except Exception as e:
                response_payload["external_sync"] = f"exception: {str(e)}"

        return jsonify(response_payload)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
