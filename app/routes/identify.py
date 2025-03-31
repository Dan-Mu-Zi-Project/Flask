from flask import Blueprint, request, jsonify
from flasgger import swag_from
import requests
import json
import os
from scipy.spatial.distance import cosine
from app.utils.api_helpers import build_external_url
from app.utils.face_utils import extract_face_embedding
from app.utils.timer import time_function
import traceback

identify_bp = Blueprint("identify_bp", __name__)

END_POINT = "/shareGroups/{shareGroupId}/vector"
COSINE_THRESHOLD = 0.35

@identify_bp.route("/identify", methods=["POST"])
@swag_from(os.path.join(os.path.dirname(__file__), "../../docs/identify.yml"))
@time_function("Identify")
def identify_person():
    if "image" not in request.files or "shareGroupId" not in request.form:
        return jsonify({"success": False, "error": "Missing image or shareGroupId"}), 400

    file = request.files["image"]
    share_group_id = str(int(request.form["shareGroupId"]))
    raw_auth_header = request.headers.get("Authorization")

    if not raw_auth_header:
        return jsonify({"success": False, "error": "Missing access token"}), 401

    access_token = raw_auth_header.replace("Bearer", "").strip()
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    if not file.mimetype.startswith("image/"):
        return jsonify({"success": False, "error": "Invalid file type"}), 400

    try:
        face_data = extract_face_embedding(file.read())

        if not isinstance(face_data, list) or not isinstance(face_data[0], dict):
            return jsonify({"success": False, "error": "Invalid face data format"}), 500

        if not face_data or len(face_data) == 0:
            return jsonify({"success": False, "error": "No face detected"}), 400

        query_vector = face_data[0]["embedding"]

        resolved_path = END_POINT.replace("{shareGroupId}", share_group_id)
        EXTERNAL_API_URL = build_external_url(resolved_path)

        response = requests.get(EXTERNAL_API_URL, headers=headers)
        if response.status_code != 200:
            return jsonify({
                "success": False,
                "error": f"Failed to fetch vector list: {response.status_code}",
                "detail": response.text
            }), 500

        members = response.json().get("data", {}).get("memberEmbeddingList", [])
        best_match = None
        min_distance = float("inf")

        for member in members:
            for item in member.get("embeddingVectorList", []):
                vector_str = item.get("vector")
                if not vector_str:
                    continue
                try:
                    candidate_vector = json.loads(vector_str)
                    distance = cosine(query_vector, candidate_vector)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = {
                            "memberId": member.get("memberId"),
                            "profileId": member.get("profileId"),
                            "name": member.get("name"),
                            "distance": round(distance, 4)
                        }
                except Exception:
                    continue

        if best_match and min_distance < COSINE_THRESHOLD:
            return jsonify({
                "success": True,
                "identified": True,
                "matched": best_match
            })
        else:
            return jsonify({
                "success": True,
                "identified": False,
                "message": "No match found within threshold",
                "min_distance": round(min_distance, 4)
            })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
