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

feedback_bp = Blueprint("feedback", __name__)
client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

def get_prompt():
    return (
      """
        너는 지금 친구들 사진을 찍어주는 중이야. 사진사 포지션이지.
        반말로, 친근하게 단체사진을 평가해 줘.\n\n
        아래 이미지를 분석한 결과를 아래 JSON 구조에 맞춰 오직 JSON 객체로만 출력해야해.
        특히 'suggestions' 항목에는 직관적이고 친근한 문장으로, 딱 1개의 조언을 짧게 해줘.
        구도에 대한 조언만 해줘. 구도 이외 통제할 수 없는 조명이나, 배경에 대한 얘긴 하지 마.\n\n
        3번~10번에 대해서 각각 1개의 조언만 도출이 될 텐데, 이것들을 토너먼트 형식으로 한번 더 모아서 제일 시급한 조언을 선정할 거야.
        3번~10번 기준을 가지고 사진을 평가했을 때 (모든 것을 상대적인 픽셀로 평가해서) 3번~10번 각각의 점수를 매겨봐. (즉, 8개의 점수)
        그리고, 네가 생각하기에 제일 시급한(제일 심각한) 번호의 조언을 딱 1개의 조언으로 선정해서 말해줘.
        다만 8번, 9번, 10번의 기준에 가중치를 조금 높게 두어줘. (무조건 우선순위로 두라는 말은 아니야.) \n\n
          
        추가로, '어떤 index를 가진 사람에게 ~를 해줘' 라는 말은, "맨 왼쪽에 있는 너! 한 발짝만 왼쪽으로 와줘"  등으로 말하듯이 매우 자연스럽게 요청해달란 이야기야.
        모든 요청은 친구가 사진을 찍어둘 때, 구도를 잡아주듯 자연스럽게 요청해 줘. 짧은 위트를 붙여도 좋겠어!
    
    
    
        
        1) face_count (int): 사진 속 사람(얼굴)의 총 개수\n
        2) face_details (array): 각 인물별 인덱스\n
           - 왼쪽부터 순서대로 인덱스 번호를 지정할 거야. (0번 부터~n번 까지)\n
           - 인덱스 번호를 말할 때의 규칙을 알려줄게.
             a. 0번 인덱스: 맨 왼쪽에 있는 너
             b. 1번 인덱스: 왼쪽에서 두 번째로 있는 사람!
             ....
             z. 마지막 인덱스: 맨 오른쪽에 있는 사람
           - 마지막 인덱스로 인식된 사람은 무조건 '맨 오른쪽에 있는 사람'이라고 불러줘!
             
           - 이런 식으로, 맨 왼쪽에서 몇 번째에 있는 사람 / 맨 오른쪽에서 몇 번째에 있는 사람 으로 불러줘.
           
        
        3) composition_1 (string): 단체사진에 나온 사람들의 얼굴을 한번에 묶은 box를 화면의 전체 폭 screenWidth와 비교해서 아래의 보기 중 1개만 선택해.\n
            즉, 해당 box를 배경과 비교해서 leftMargin과 rightMargin 값을 가지고 아래를 평가해줘.\n
            3-1) 사람들이 화면 기준 왼쪽으로 치우쳐져 있다면, 너무 왼쪽으로 치우쳐 있으니 오른쪽으로 조금만 가달라고 말해줘.\n
            3-2) 사람들이 화면 기준 오른쪽으로 치우쳐져 있다면, 너무 오른쪽으로 치우쳐 있으니 왼쪽으로 조금만 가달라고 말해줘.\n
            3-3) LeftMargin과 rightMargin이 너무 작으면, 모두가 너무 좌우 끝까지 퍼져 있으니 조금 붙어 서 달라고 말해줘.\n
            이 보기들은 '꽤 치우쳐져 있을 때'를 말한 거야. 엄격하게 하지 말고, 적당히 괜찮으면 패스해도 돼.\n
        4) composition_2 (string): 개별 얼굴 간의 거리를 분석해서, 아래 기준으로 평가해줘. 아래 보기 중 1개만 선택해.\n
            즉, 각 사람들의 얼굴 box마다 가로 방향 거리 (옆 사람과의 간격) 를 가지고 아래를 평가해줘.\n
            4-1) 다른 간격들보다 유독 먼 간격을 가진 얼굴이 있다면, 해당 index를 가진 사람에게 조금만 붙어달라고 해줘.\n
            4-2) 다른 간격들보다 유독 가까운 간격을 가진 얼굴이 있다면, 해당 index를 가진 사람에게 조금만 떨어져 달라고 해줘.\n
            이 보기들은 '꽤 멀거나 가까울 때'를 말한 거야. 엄격하게 하지 말고, 적당히 괜찮으면 패스해도 돼.\n
        5) composition_3 (string): 개별 얼굴 간의 거리를 분석해서, 아래 기준으로 평가해줘. 아래 보기 중 1개만 선택해.\n
            즉, 각 사람들의 얼굴 box마다 얼굴 간의 상하 거리 (즉, 키 차이) 를 가지고 아래를 평가해줘.\n
            5-1) 다른 간격들보다 유독 높은 얼굴이 있다면, 해당 index를 가진 사람에게 키가 크니 가운데로 와달라고 해줘.\n
            5-2) 다른 간격들보다 유독 낮은 간격을 가진 얼굴이 있다면, 해당 index를 가진 사람에게 가장자리로 가달라고 해줘.\n
            이 보기들은 '꽤 높거나 낮을 때'를 말한 거야. 엄격하게 하지 말고, 적당히 괜찮으면 패스해도 돼.\n
            가운데에 있는 사람에게는 또 가운데로 오라고 말하지 마.
        6) composition_4 (string): 전체 사람들의 좌우 여백을 체크해서, 좌우 여백이 너무 작다면, 좌우 여백이 좁으니 조금만 붙어 달라고 해줘.
            이 보기들은 '꽤 여백이 적을 때'를 말한 거야. 엄격하게 하지 말고, 적당히 괜찮으면 패스해도 돼.\n
        7) composition_5 (string): 사람들의 위쪽 여백을 체크해서, 위쪽 여백이 너무 적으면, 조금 뒤로 물러나거나 카메라를 위로 조정해 달라고 해줘.\n
            이것은 '꽤 여백이 적을 때'를 말한 거야. 엄격하게 하지 말고, 적당히 괜찮으면 패스해도 돼.\n
        8) composition_6 (string): 프레임에서 벗어나는 사람이 있다면, 모두 프레임 안쪽으로 들어와 정면을 봐달라고 해줘.\n
            이것은 '꽤 겹칠 때'를 말한 거야. 엄격하게 하지 말고, 적당히 괜찮으면 패스해도 돼.\n
        9) composition_7 (string): Bounding Box 겹침 비율을 기준으로, 일정 이상 겹치면, 해당 index를 가진 사람에게 얼굴이 가려져 있으니 떨어져 달라고 해줘.\n
            만약 앞줄/뒷줄 구분이 있을 시에는, 뒷줄 사람의 얼굴이 보이지 않는다면, 뒷줄 얼굴이 안 보이니 얼굴을 보여 달라고 해줘.\n
            이것은 '꽤 겹칠 때'를 말한 거야. 엄격하게 하지 말고, 적당히 괜찮으면 패스해도 돼.\n
        10) composition_8 (string): 만약 얼굴의 bounding box의 크기가 다르다면, 
            작은 bounding box를 가진 사람에게 (즉, 멀리 있는 사람에게) 앞으로 와달라고 해줘.
            이것은 '꽤 멀리 있을 때'를 말한거야. 엄격하게 하지 말고, 적당히 괜찮으면 패스해도 돼.\n
        11) suggestions (string): 1개의 친근한 조언\n\n
        예시:\n
        {\n"
          \"face_count\": 5,\n
          \"face_details\": [1, 2, 3, 4, 5], \n
          \"suggestions\": \"왼쪽에서 두 번째에 있는 너! 키가 참 크다. 그러니 가운데로 와줘!\"\n
        }\n\n
        그 외 설명이나 문장은 넣지 말고, 오직 JSON 객체만 반환해 줘.
        각 composition에 대해 평가한 내용도 "composition_" 항목에서 알려줘.
        """
    )
    
    


@feedback_bp.route("/feedback", methods=["POST"])
@swag_from(os.path.join(os.path.dirname(__file__), "../../../docs/feedback.yml"))
def photo_feedback():
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
        parsed = json.loads(cleaned_text)

        tts_text = parsed["suggestions"].replace("\n", " ")

        speech_path = Path(tempfile.mktemp(suffix=".mp3"))
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=tts_text
        ) as speech_response:
            speech_response.stream_to_file(speech_path)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "처리 실패", "message": str(e)}), 500

    response = send_file(speech_path, mimetype="audio/mpeg", as_attachment=False)

    @response.call_on_close
    def cleanup():
        try:
            speech_path.unlink()
        except:
            pass

    return response