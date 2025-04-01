from flask import Blueprint, request, jsonify
from flasgger import swag_from
import os
import traceback

from app.utils.face_utils import extract_multiple_face_embeddings
from app.utils.upload_utils import request_presigned_url

upload_photo_bp = Blueprint("debug_bp", __name__)


@upload_photo_bp.route("/face/debug/upload_photo", methods=["POST"])
@swag_from(os.path.join(os.path.dirname(__file__), "../../docs/upload_photo.yml"))
def upload_photo():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image part in the request"}), 400

        photo_name = request.form.get("photoName")
        if not photo_name:
            return jsonify({"error": "Missing photoName in form data"}), 400

        auth_header = request.headers.get("Authorization")
        access_token = auth_header.replace("Bearer ", "").strip() if auth_header else None

        image_file = request.files["image"]
        image_bytes = image_file.read()

        result = extract_multiple_face_embeddings(image_bytes)

        valid_embeddings = [
            r for r in result if "embedding" in r and isinstance(r["embedding"], list)
        ]

        presigned_url_response = None

        if valid_embeddings:
            presigned_url_response = request_presigned_url([photo_name], access_token)

        return jsonify({
            "result": result,
            "presigned_url_response": presigned_url_response
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
