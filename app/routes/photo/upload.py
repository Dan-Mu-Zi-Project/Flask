from flask import Blueprint, request, jsonify
from flasgger import swag_from
import os
import traceback
import uuid

from app.utils.face_utils import extract_multiple_face_embeddings, find_best_match
from app.utils.upload_utils import request_group_face_vectors, request_presigned_url, upload_image_to_presigned_url, finalize_photo_upload

upload_bp = Blueprint("debug_bp", __name__)

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

        if not share_group_id or not location or not take_at:
            return jsonify({"error": "Missing required form fields"}), 400

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        access_token = auth_header.replace("Bearer ", "").strip()

        result = extract_multiple_face_embeddings(image_bytes)
        valid_embeddings = [r for r in result if "embedding" in r]
        # if not valid_embeddings:
        #     return jsonify({"error": "No valid face embeddings detected"}), 400

        query_embedding = valid_embeddings[0]["embedding"]
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

        upload_image_to_presigned_url(presigned_url, image_bytes)

        upload_result = finalize_photo_upload(
            share_group_id=share_group_id,
            photo_url=photo_url,
            profile_id_list=profile_id_list,
            location=location,
            taked_at=take_at,
            image_bytes=image_bytes,
            access_token=access_token
        )

        return jsonify({
            "message": "✅ 사진 업로드 및 등록 완료",
            "matchedProfiles": best_matches,
            "imageUrl": photo_url,
            "uploadResult": upload_result
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500