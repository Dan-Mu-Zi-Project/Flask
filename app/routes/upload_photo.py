from flask import Blueprint, request, jsonify
from flasgger import swag_from
import os
import traceback

from app.utils.face_utils import extract_multiple_face_embeddings

upload_photo_bp = Blueprint("debug_bp", __name__)


@upload_photo_bp.route("/face/debug/upload_photo", methods=["POST"])
@swag_from(os.path.join(os.path.dirname(__file__), "../../docs/upload_photo.yml"))
def upload_photo():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image part in the request"}), 400

        image_file = request.files["image"]
        image_bytes = image_file.read()

        result = extract_multiple_face_embeddings(image_bytes)

        return jsonify({"result": result}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
