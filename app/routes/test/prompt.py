from flask import Blueprint, request, jsonify, send_file, Response
from flasgger import swag_from
import openai
import base64
import json
import os
import tempfile
import traceback
from pathlib import Path
from app.config import Config

prompt_bp = Blueprint("prompt", __name__)
client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

def get_prompt():
    return (
        "당신은 10년 경력의 사진평론가 겸 사진작가입니다. "
        "하지만 지나치게 딱딱한 표현보다는, 친절하고 이해하기 쉬운 말투로 단체사진을 평가해 주세요.\n\n"
        "아래 이미지를 분석한 결과를 아래 JSON 구조에 맞춰 오직 JSON 객체로만 출력해 주십시오. "
        "특히 'suggestions' 항목에는 짧고 직관적인 문장으로, 2~3개의 조언을 Bullet Point 형태로 적어주세요. "
        "나머지 항목에도 전문용어가 필요하면 영어(Korean (English)) 표기나 간단한 설명을 병기해 주세요.\n\n"
        "1) face_count (int): 사진 속 사람(얼굴)의 총 개수\n"
        "2) face_details (array): 각 인물별 상세 정보\n"
        "   - index (int): 왼쪽부터 0, 1, 2...\n"
        "   - eyes_closed (bool): 눈을 완전히/반쯤 감았으면 True\n"
        "   - focus_ok (bool): 초점이 잘 맞았으면 True, 흐리면 False\n"
        "3) composition (string): 구도, 인물 배치 등 평가\n"
        "4) lighting_evaluation (string): 조명/노출 상태 평가\n"
        "5) overall_score (int): 단체사진 종합 점수 (0~100)\n"
        "6) suggestions (string): Bullet Point 형태로 2~3개 친근한 조언\n\n"
        "예시:\n"
        "{\n"
        "  \"face_count\": 5,\n"
        "  \"face_details\": [\n"
        "    {\"index\": 0, \"eyes_closed\": false, \"focus_ok\": true},\n"
        "    {\"index\": 1, \"eyes_closed\": true, \"focus_ok\": false}\n"
        "  ],\n"
        "  \"composition\": \"가운데 인물 중심의 Three-point composition.\",\n"
        "  \"lighting_evaluation\": \"자연광이 은은하지만 배경이 약간 어두움.\",\n"
        "  \"overall_score\": 85,\n"
        "  \"suggestions\": \"- 조금 더 밝은 조명을 사용해 보세요\\n- 카메라 흔들림을 줄여보세요\"\n"
        "}\n\n"
        "그 외 설명이나 문장은 넣지 마시고, 오직 JSON 객체만 반환해 주세요."
    )

@prompt_bp.route("/prompt", methods=["POST"])
@swag_from(os.path.join(os.path.dirname(__file__), "../../../docs/prompt.yml"))
def photo_feedback_debug():
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

        user_input = [
            {"type": "input_text", "text": get_prompt()},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64_image}"}
        ]

        gpt_response = client.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": user_input}]
        )

        text_content = gpt_response.output[0].content[0].text
        cleaned_text = text_content.strip().removeprefix("```json").removesuffix("```").strip()

        return jsonify({
            "success": True,
            "feedback": cleaned_text
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
