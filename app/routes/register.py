from flask import Blueprint, request, jsonify
from flasgger import swag_from
import numpy as np
import os
import cv2
import json
import requests
import time
from app.utils.api_helpers import build_external_url
from app.utils.face_utils import extract_face_embedding
from app.utils.timer import time_function

register_buffer = {}

GC_TIMEOUT_SECONDS = 90
FACE_CONFIDENCE_THRESHOLD = 0.82
REQUIRED_ANGLES = {"FRONT", "LEFT", "RIGHT"}
END_POINT = "/members/sampleImage"

register_bp = Blueprint("register_bp", __name__)


@register_bp.route("/register", methods=["POST"])
@swag_from(os.path.join(os.path.dirname(__file__), "../../docs/register.yml"))
@time_function("Register")
def register():
    gc_expired_entries()
    EXTERNAL_API_URL = build_external_url(END_POINT)

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400

    file = request.files["image"]
    angle_type = request.form.get("angle_type", "").strip().upper()

    if angle_type not in REQUIRED_ANGLES:
        return (
            jsonify({"success": False, "error": "Invalid or missing angle_type"}),
            400,
        )

    auth_header = request.headers.get("Authorization")

    if not auth_header:
        return (
            jsonify({"success": False, "error": "Missing or invalid access token"}),
            401,
        )

    access_token = auth_header.replace("Bearer ", "").strip()
    user_token_id = access_token  # 메모리 기반 버퍼의 키로 사용

    if not file.mimetype.startswith("image/"):
        return jsonify({"success": False, "error": "Invalid file type."}), 400

    try:
        face_data = extract_face_embedding(file.read())

        face_confidence = face_data[0].get("face_confidence", 0.0)
        if face_confidence < FACE_CONFIDENCE_THRESHOLD:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Face confidence too low: {face_confidence:.4f}",
                    }
                ),
                400,
            )

        embedding = face_data[0]["embedding"]
        face_vector = json.dumps(embedding)

        if user_token_id not in register_buffer:
            register_buffer[user_token_id] = {"_created_at": time.time()}

        register_buffer[user_token_id][angle_type] = {
            "angleType": angle_type,
            "faceVector": face_vector,
        }

        response_payload = {
            "success": True,
            "angle_type": angle_type,
            "registered_angles": list(register_buffer[user_token_id].keys()),
            "all_angles_completed": False,
        }

        if REQUIRED_ANGLES.issubset(register_buffer[user_token_id].keys()):
            try:
                face_sample_list = [
                    data
                    for angle, data in register_buffer[user_token_id].items()
                    if not angle.startswith("_")
                ]

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {access_token}",
                }

                payload = {"faceSampleList": face_sample_list}
                response = requests.post(
                    EXTERNAL_API_URL, data=json.dumps(payload), headers=headers
                )

                if response.status_code == 200:
                    response_payload["all_angles_completed"] = True
                    response_payload["external_sync"] = "success"
                    register_buffer.pop(
                        user_token_id, None
                    )  # 외부 전송 성공 시 메모리 정리
                else:
                    response_payload["external_sync"] = (
                        f"failed: {response.status_code} - {response.text}"
                    )

            except Exception as e:
                response_payload["external_sync"] = f"exception: {str(e)}"

        return jsonify(response_payload)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def gc_expired_entries():
    now = time.time()
    expired_keys = []

    for user_token_id, data in register_buffer.items():
        created_at = data.get("_created_at", now)
        if now - created_at > GC_TIMEOUT_SECONDS:
            expired_keys.append(user_token_id)

    for key in expired_keys:
        register_buffer.pop(key, None)


@register_bp.route("/register/debug", methods=["GET"])
def debug_register_buffer():
    from datetime import datetime

    debug_data = {}
    for user_token_id, user_data in register_buffer.items():
        created_at_ts = user_data.get("_created_at")
        created_at_str = (
            datetime.fromtimestamp(created_at_ts).isoformat()
            if created_at_ts
            else "N/A"
        )

        angle_data = {k: v for k, v in user_data.items() if not k.startswith("_")}

        debug_data[user_token_id] = {
            "created_at": created_at_str,
            "angles_registered": list(angle_data.keys()),
        }

    return jsonify(
        {"success": True, "buffer_summary": debug_data, "total_users": len(debug_data)}
    )
