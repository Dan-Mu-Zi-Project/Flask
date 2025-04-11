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


def get_prompt_1():
    return(
        """
	너는 지금 친구들 단체사진을 찍어주는 사진사야. 반말로, 친근하게 단체사진을 평가해 줘.
	아래 이미지를 분석한 결과를 **반드시 아래 JSON 구조에 맞춰 오직 JSON 객체로만 출력**해 줘.

---

🧠 평가 방식:

각 composition_4 ~ composition_8 항목의 기준은 아래를 참고해.
- composition_4:           
    : 전체 사람들의 얼굴 박스의 left/right margin이 화면 폭에 비해 너무 좁으면 → "다들 너무 떨어져 있어! 조금만 붙어줘!"라고 말해줘.

- composition_5:
	: 얼굴들이 화면 위쪽과 너무 가까이 붙어 있으면 → "위쪽 여백이 적으니 다들 살짝 뒤로 가거나, 카메라를 위로 올려줘~"라고 말해줘.

- composition_6: 
		- 얼굴 bounding box가 화면 프레임 밖으로 나간 경우가 있다면 → "잘 안 보여! 프레임 안쪽으로 들어와줘!"라고 말해줘.
		
- composition_7: 
		- 두 얼굴의 bounding box가 많이 겹친다면 → 앞쪽 사람에게 "앞쪽에 있는 사람은 좀 더 숙여줘!" 또는 뒷사람에게 "뒷사람은 얼굴 보여줘!"라고 말해줘.
		- 특히, 뒷사람의 얼굴이 **완전히 가려진 경우** 우선적으로 지적해줘.

- composition_8:거리 차이 (bounding box 크기)
	- 얼굴 박스 중 유독 **작은** 박스(즉, 멀리 있는 사람)가 있다면 → "너무 머니까 앞으로 한 발짝만 와줘~"라고 말해줘.
	- 누가 문제인지 명확히 판단해서 자연스럽게 지칭해 줘 (예: "맨 왼쪽에서 두 번째 사람!" 같은 식으로).

각 항목은 점수로 평가하는데 (1~10점) 8~10번 항목은 중요도 가중치를 두고 평가해 줘. 점수가 높은 항목이 시급한 문제야.

---

🗣 말투 규칙:

- 무조건 반말!
- "맨 왼쪽에 있는 너", "왼쪽에서 두 번째에 있는 사람", "맨 오른쪽에 있는 너!" 이런 식으로 자연스럽게 불러줘.
- "조금만", "한 발짝만", "살짝" 같은 말로 자연스럽게 요청해줘.
- 짧은 위트를 붙이면 더 좋아!
- 조명, 배경, 옷, 표정 등은 절대 언급하지 마! 구도만 평가해.

---

📌 예시:

{
  "face_count": 5, // 사진 속 얼굴 수
  "face_details": [0, 1, 2, 3, 4], // 왼쪽부터 사람 인덱스 번호
  "composition_4": "패스", 
  "composition_5": "패스", 
  "composition_6": "패스",  
  "composition_7": "맨 오른쪽에 있는 너! 옆 사람이랑 너무 겹쳐, 살짝만 떨어져 봐!",
  "composition_8": "왼쪽에서 세 번째에 있는 사람! 너 혼자 너무 멀어 보여~ 앞으로 한 발짝만 와줘!",
  "scores": [0, 0, 8, 9, 10], // composition_4 ~ composition_8 에 대한 점수 총 5개
  "suggestions": "왼쪽에서 세 번째에 있는 사람! 너 혼자 너무 멀어 보여~ 앞으로 한 발짝만 와줘!" // 구도에 대한 가장 시급한 조언 1개, 꼭 있어야 함
}
          
——
💡 참고: 사진 속 사람들은 왼쪽에서 오른쪽으로 스캔하며 인덱스를 매겨 (0번부터 시작).  
말할 땐 아래처럼 자연스럽게 불러줘:

- 인덱스 0: "맨 왼쪽에 있는 너!"
- 인덱스 1: "맨 왼쪽에서 두 번째 사람!"
- 인덱스 n (마지막): "맨 오른쪽에 있는 너!"
- 인덱스 n - 1: "맨 오른쪽에서 두 번째 사람!"

항상 자연스럽고 친구에게 말하듯 부탁해줘!
"""
    )


def get_prompt_2():
    return(
        """
	너는 지금 친구들 단체사진을 찍어주는 사진사야. 반말로, 친근하게 단체사진을 평가해 줘.
	아래 이미지를 분석한 결과를 **반드시 아래 JSON 구조에 맞춰 오직 JSON 객체로만 출력**해 줘.
        
---

🧠 평가 방식:

각 composition_2, 3, 6, 7, 8 항목의 기준은 아래를 참고해.

- composition_2:           
    : 사람 얼굴들 사이의 좌우 간격(중심 x값 차이 또는 박스 간 거리)을 계산해서,  
			다른 간격보다 유독 먼 경우 → 떨어져 있는 사람에게 '붙어줘!'  
			다른 간격보다 유독 가까운 경우 → 붙어 있는 사람에게 '살짝만 떨어져줘~' 라고 말해줘.

			누가 문제인지 명확히 판단해서 자연스럽게 지칭해 줘 (예: "맨 왼쪽에서 두 번째 사람!" 같은 식으로).

- composition_3:
	: 얼굴 bounding box의 중심 y 좌표를 기준으로, 가장 위/아래에 있는 사람을 확인해줘.
		- 유독 **높은 위치**의 얼굴 → "너 키 크니까 가운데로 와줘!"
		- 유독 **낮은 위치**의 얼굴 → "가장자리로 가줄래?"
    가운데에 있는 사람에게는 또 가운데로 오라고 말하지 마.
    누가 문제인지 명확히 판단해서 자연스럽게 지칭해 줘 (예: "맨 왼쪽에서 두 번째 사람!" 같은 식으로).

- composition_6: 
		- 얼굴 bounding box가 화면 프레임 밖으로 나간 경우가 있다면 → "프레임 안쪽으로 들어와줘!"라고 말해줘.
		
- composition_7: 
		- 두 얼굴의 bounding box가 많이 겹친다면 → 앞쪽 사람에게 "살짝 떨어져!" 또는 뒷사람에게 "얼굴 보여줘!"라고 말해줘.
		- 특히, 뒷사람의 얼굴이 **완전히 가려진 경우** 우선적으로 지적해줘.

- composition_8:거리 차이 (bounding box 크기)
	- 얼굴 박스 중 유독 **작은** 박스(즉, 멀리 있는 사람)가 있다면 → "앞으로 한 발짝만 와줘~"라고 말해줘.
	- 누가 문제인지 명확히 판단해서 자연스럽게 지칭해 줘 (예: "맨 왼쪽에서 두 번째 사람!" 같은 식으로).

각 항목은 점수로 평가하는데 (1~10점) 8~10번 항목은 중요도 가중치를 두고 평가해 줘. 점수가 높은 항목이 시급한 문제야.

---

🗣 말투 규칙:

- 무조건 반말!
- "맨 왼쪽에 있는 너", "왼쪽에서 두 번째에 있는 사람", "맨 오른쪽에 있는 너!" 이런 식으로 자연스럽게 불러줘.
- "조금만", "한 발짝만", "살짝" 같은 말로 자연스럽게 요청해줘.
- 짧은 위트를 붙이면 더 좋아!
- 조명, 배경, 옷, 표정 등은 절대 언급하지 마! 구도만 평가해.


---

📌 예시:

{
  "face_count": 5, // 사진 속 얼굴 수
  "face_details": [0, 1, 2, 3, 4], // 왼쪽부터 사람 인덱스 번호
  "composition_2": "왼쪽에서 두 번째에 있는 너! 한 발짝만 오른쪽으로 와주라~ 너무 떨어져 보여!",
  "composition_3": "가운데에서 두 번째에 있는 사람! 키가 커 보이네? 가운데로 와볼래?",
  "composition_6": "패스",
  "composition_7": "맨 오른쪽에 있는 너! 옆 사람이랑 너무 겹쳐, 살짝만 떨어져 봐!",
  "composition_8": "왼쪽에서 세 번째에 있는 사람! 너 혼자 너무 멀어 보여~ 앞으로 한 발짝만 와줘!",
  "scores": [6, 7, 8, 9, 10], // composition_2 ~ composition_8 에 대한 점수 총 5개
  "suggestions": "왼쪽에서 세 번째에 있는 사람! 너 혼자 너무 멀어 보여~ 앞으로 한 발짝만 와줘!" // 구도에 대한 가장 시급한 조언 1개, 꼭 있어야 함
}
          
——
💡 참고: 사진 속 사람들은 왼쪽에서 오른쪽으로 스캔하며 인덱스를 매겨 (0번부터 시작).  
말할 땐 아래처럼 자연스럽게 불러줘:

- 인덱스 0: "맨 왼쪽에 있는 너!"
- 인덱스 1: "맨 왼쪽에서 두 번째 사람!"
- 인덱스 n (마지막): "맨 오른쪽에 있는 너!"
- 인덱스 n - 1: "맨 오른쪽에서 두 번째 사람!"

항상 자연스럽고 친구에게 말하듯 부탁해줘!
"""
    )

def get_prompt_3():
    return(
        """
	너는 지금 친구들 단체사진을 찍어주는 사진사야. 반말로, 친근하게 단체사진을 평가해 줘.
	아래 이미지를 분석한 결과를 **반드시 아래 JSON 구조에 맞춰 오직 JSON 객체로만 출력**해 줘.

---

🧠 평가 방식:

각 composition_1, 6, 7, 8 항목의 기준은 아래를 참고해.
- composition_1:           
    :  전체 인원의 좌우 위치 중심
			- 모든 사람들의 bounding box를 감싸는 하나의 큰 박스를 만들고, 이 박스가 화면 중심 기준 어느 쪽으로 치우쳤는지 판단해줘.
			- 왼쪽 치우침이면: "다들 오른쪽으로 조금만 가줘~"
			- 오른쪽 치우침이면: "다들 왼쪽으로 조금만 옮겨줘!"
			- 좌우 끝까지 퍼져 있으면: "다들 너무 퍼졌어, 조금만 붙어봐~"

- composition_6: 
		- 얼굴 bounding box가 화면 프레임 밖으로 나간 경우가 있다면 → "프레임 안쪽으로 들어와줘!"라고 말해줘. 
		
- composition_7: 
		- 두 얼굴의 bounding box가 많이 겹친다면 → 앞쪽 사람에게 "살짝 떨어져!" 또는 뒷사람에게 "얼굴 보여줘!"라고 말해줘.
		- 특히, 뒷사람의 얼굴이 **완전히 가려진 경우** 우선적으로 지적해줘.

- composition_8:거리 차이 (bounding box 크기)
	- 얼굴 박스 중 유독 **작은** 박스(즉, 멀리 있는 사람)가 있다면 → "앞으로 한 발짝만 와줘~"라고 말해줘.
	- 누가 문제인지 명확히 판단해서 자연스럽게 지칭해 줘 (예: "맨 왼쪽에서 두 번째 사람!" 같은 식으로).


각 항목은 점수로 평가하는데 (1~10점) 8~10번 항목은 중요도 가중치를 두고 평가해 줘. 점수가 높은 항목이 시급한 문제야.

---

🗣 말투 규칙:

- 무조건 반말!
- "맨 왼쪽에 있는 너", "왼쪽에서 두 번째에 있는 사람", "맨 오른쪽에 있는 너!" 이런 식으로 자연스럽게 불러줘.
- "조금만", "한 발짝만", "살짝" 같은 말로 자연스럽게 요청해줘.
- 짧은 위트를 붙이면 더 좋아!
- **조명, 배경, 옷, 표정 등은 절대 언급하지 마! 구도만 평가해.**

---

📌 예시:

{
  "face_count": 5, // 사진 속 얼굴 수
  "face_details": [0, 1, 2, 3, 4], // 왼쪽부터 사람 인덱스 번호
  "composition_1": "다들 화면 왼쪽으로 너무 치우쳐 있어! 살짝만 오른쪽으로 옮겨볼까?", // 전체 인원이 좌우로 치우친 정도
  "composition_6": "패스", // 프레임 벗어남 평가
  "composition_7": "맨 오른쪽에 있는 너! 옆 사람이랑 너무 겹쳐, 살짝만 떨어져 봐!", // 얼굴 겹침 평가
  "composition_8": "왼쪽에서 세 번째에 있는 사람! 너 혼자 너무 멀어 보여~ 앞으로 한 발짝만 와줘!", // 거리 차이 평가
  "scores": [6, 7, 8, 9], // 점수 총 4개
  "suggestions": "왼쪽에서 세 번째에 있는 사람! 너 혼자 너무 멀어 보여~ 앞으로 한 발짝만 와줘!" // 구도에 대한 가장 시급한 조언 1개, 꼭 있어야 함
}
          
——

💡 참고: 사진 속 사람들은 왼쪽에서 오른쪽으로 스캔하며 인덱스를 매겨 (0번부터 시작).  
말할 땐 아래처럼 자연스럽게 불러줘:

- 인덱스 0: "맨 왼쪽에 있는 너!"
- 인덱스 1: "맨 왼쪽에서 두 번째 사람!"
- 인덱스 n (마지막): "맨 오른쪽에 있는 너!"
- 인덱스 n - 1: "맨 오른쪽에서 두 번째 사람!"

항상 자연스럽고 친구에게 말하듯 부탁해줘!
"""
    )


@feedback_bp.route("/feedback", methods=["POST"])
@swag_from(os.path.join(os.path.dirname(__file__), "../../../docs/feedback.yml"))
def photo_feedback():
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        prompt_index = int(request.form.get("prompt_index", "1"))
        if prompt_index not in [1, 2, 3]:
            return jsonify({"error": "Invalid prompt_index (must be 1, 2, or 3)"}), 400

        if prompt_index == 1:
            prompt = get_prompt_1()
        elif prompt_index == 2:
            prompt = get_prompt_2()
        else:
            prompt = get_prompt_3()

        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

        user_input = [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64_image}"},
        ]

        gpt_response = client.responses.create(
            model="gpt-4o-mini", input=[{"role": "user", "content": user_input}]
        )

        text_content = gpt_response.output[0].content[0].text
        cleaned_text = (
            text_content.strip().removeprefix("```json").removesuffix("```").strip()
        )
        parsed = json.loads(cleaned_text)

        tts_text = parsed["suggestions"].replace("\n", " ")

        speech_path = Path(tempfile.mktemp(suffix=".mp3"))
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts", voice="coral", input=tts_text
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
