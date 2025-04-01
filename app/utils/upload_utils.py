import requests

from app.utils.api_helpers import build_external_url

def request_presigned_url(photo_name_list, access_token=None):
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
            headers=headers
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
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        print(f"[DEBUG] 응답 코드: {response.status_code}")
        print(f"[DEBUG] 응답 내용: {response.text}")
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"GET /shareGroups/{share_group_id}/vector 요청 실패: {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"공유 그룹 벡터 요청 중 예외 발생: {str(e)}")
    
