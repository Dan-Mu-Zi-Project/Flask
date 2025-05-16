from flask import Blueprint, request, jsonify, current_app
from flasgger import swag_from
import base64
from openai import OpenAI
import os
import time

# 환경 변수 로드 (만약 .env 파일이 프로젝트 루트에 있다면, app/__init__.py 등에서 이미 로드되었을 수 있습니다.)
# from dotenv import load_dotenv
# load_dotenv()

# OpenAI 클라이언트 초기화
# API 키는 환경 변수에서 가져옵니다. Flask 설정에서 관리하는 것이 더 일반적입니다.
# client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) 
# 이 부분은 Flask 앱의 설정에서 OpenAI 클라이언트를 초기화하고 Blueprint에서 사용할 수 있도록 수정해야 할 수 있습니다.
# 우선은 여기에 두겠습니다.

image_analysis_bp = Blueprint('image_analysis', __name__, url_prefix='/test')

# Function to encode the image
def encode_image_from_file_storage(file_storage):
    return base64.b64encode(file_storage.read()).decode("utf-8")

@image_analysis_bp.route('/analyze', methods=['POST'])
@swag_from({
    'tags': ['Image Analysis'],
    'summary': 'Analyze an image to provide suggestions for better composition.',
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'image',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'The image file to analyze (e.g., aespa.jpg).'
        },
        {
            'name': 'prompt',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'default': '왼쪽부터[카리나, 지젤, 닝닝, 윈터] 각 인물이 어떻게 이동해야 사진이 더 잘 나올까. 이름과 함께 시급한 한명만 간결하게 지시해줘. 근거도 알려줘',
            'description': 'Prompt for the image analysis.'
        }
    ],
    'responses': {
        200: {
            'description': 'Image analysis result.',
            'schema': {
                'type': 'object',
                'properties': {
                    'analysis_result': {'type': 'string'},
                    'encoding_time': {'type': 'number'},
                    'api_call_time': {'type': 'number'},
                    'time_to_first_chunk': {'type': 'number', 'nullable': True},
                    'total_script_time': {'type': 'number'}
                }
            }
        },
        400: {
            'description': 'Bad request, e.g., no image file provided.'
        },
        500: {
            'description': 'Internal server error during analysis.'
        }
    }
})
def analyze_image_route():
    script_start_time = time.time()

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    custom_prompt = request.form.get('prompt', '왼쪽부터[카리나, 지젤, 닝닝, 윈터] 각 인물이 어떻게 이동해야 사진이 더 잘 나올까. 이름과 함께 시급한 한명만 간결하게 지시해줘. 근거도 알려줘')
    
    # OpenAI 클라이언트 초기화 - 앱 컨텍스트에서 가져오거나 여기서 직접 초기화
    # 실제 운영 환경에서는 Flask 앱의 config에서 API 키를 관리하고, app 컨텍스트를 통해 client를 공유하는 것이 좋습니다.
    try:
        # client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY')) # .env 파일이 루트에 있고 load_dotenv()가 호출되었다고 가정
        api_key = current_app.config.get('OPENAI_API_KEY')
        if not api_key:
            # raise ValueError("OPENAI_API_KEY is not set in environment variables.")
            current_app.logger.error("OPENAI_API_KEY is not set in the application configuration.")
            return jsonify({'error': 'OpenAI API key is not configured.'}), 500
        client = OpenAI(api_key=api_key)
    except Exception as e:
        current_app.logger.error(f'Failed to initialize OpenAI client: {str(e)}')
        return jsonify({'error': f'Failed to initialize OpenAI client: {str(e)}'}), 500

    encode_start_time = time.time()
    try:
        base64_image = encode_image_from_file_storage(image_file)
    except Exception as e:
        return jsonify({'error': f'Failed to encode image: {str(e)}'}), 500
    encode_end_time = time.time()
    encoding_time = encode_end_time - encode_start_time

    analysis_text_result = []
    first_chunk_received_time = None
    
    api_call_start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14", 
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": custom_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}", # Assuming JPEG, might need to detect type
                                "detail": "low" 
                            }
                        },
                    ],
                }
            ],
            temperature=0.0,
            top_p=1.0,
            n=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=42,
            max_tokens=150, # 늘림
        )

        for chunk in response:
            if first_chunk_received_time is None and chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                first_chunk_received_time = time.time()
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                analysis_text_result.append(chunk.choices[0].delta.content)
        
    except Exception as e:
        return jsonify({'error': f'OpenAI API call failed: {str(e)}'}), 500
    
    api_call_end_time = time.time()
    api_call_and_streaming_time = api_call_end_time - api_call_start_time
    time_to_first_chunk = (first_chunk_received_time - api_call_start_time) if first_chunk_received_time else None

    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time

    return jsonify({
        'analysis_result': "".join(analysis_text_result),
        'encoding_time': round(encoding_time, 4),
        'api_call_time': round(api_call_and_streaming_time, 4),
        'time_to_first_chunk': round(time_to_first_chunk, 4) if time_to_first_chunk is not None else None,
        'total_script_time': round(total_script_time, 4)
    })

# 여기에 더 많은 라우트나 헬퍼 함수를 추가할 수 있습니다. 