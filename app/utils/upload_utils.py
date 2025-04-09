import requests

from app.utils.api_helpers import build_external_url
import json
import cv2
from io import BytesIO
import numpy as np


def finalize_photo_upload(
    share_group_id,
    photo_url,
    profile_id_list,
    location,
    taked_at,
    image_bytes,
    access_token,
):
    url = build_external_url(f"/photos/{share_group_id}")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]

    payload = {
        "photoList": [
            {
                "imageUrl": photo_url,
                "profileIdList": profile_id_list,
                "location": location,
                "takedAt": taked_at,
                "width": width,
                "height": height,
            }
        ]
    }

    print("[DEBUG] /photos payload:", json.dumps(payload, indent=2))

    try:
        response = requests.post(url, json=payload, headers=headers)
        print("[DEBUG] /photos response:", response.status_code, response.text)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"최종 업로드 요청 실패: {str(e)}")


def upload_image_to_presigned_url(presigned_url, image_bytes):
    headers = {"Content-Type": "image/jpeg"}
    response = requests.put(presigned_url, data=image_bytes, headers=headers)
    response.raise_for_status()
    return True


def request_presigned_url(photo_name_list, access_token):
    """
    presigned URL 요청을 보내는 함수 (Bearer 토큰 지원)
    """
    END_POINT = "/photos/preSignedUrl"
    EXTERNAL_API_URL = build_external_url(END_POINT)
    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    try:
        response = requests.post(
            url=EXTERNAL_API_URL,
            json={"photoNameList": photo_name_list},
            headers=headers,
        )
        print("[DEBUG] response.text:", response.text)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise RuntimeError(f"Presigned URL 요청 실패: {str(e)}")


import requests


def request_group_face_vectors(share_group_id: int, access_token: str):
    """
    공유 그룹 ID로 얼굴 벡터 리스트 요청
    """
    END_POINT = "/shareGroups/{share_group_id}/vector"
    resolved_path = END_POINT.replace("{share_group_id}", str(share_group_id))
    EXTERNAL_API_URL = build_external_url(resolved_path)
    url = EXTERNAL_API_URL
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"GET /shareGroups/{share_group_id}/vector 요청 실패: {e.response.text}"
        )
    except Exception as e:
        raise RuntimeError(f"공유 그룹 벡터 요청 중 예외 발생: {str(e)}")

def get_current_share_group_id(token: str) -> int:
    END_POINT = "/shareGroups/current"
    EXTERNAL_API_URL = build_external_url(END_POINT)
    url = EXTERNAL_API_URL
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    json_data = response.json()
    
    if "data" not in json_data or "shareGroupId" not in json_data["data"]:
        raise ValueError("Invalid response format: missing shareGroupId")


    return json_data["data"]["shareGroupId"]