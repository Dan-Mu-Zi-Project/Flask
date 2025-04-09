from flask import Blueprint, request, jsonify, send_file
from flasgger import swag_from
import base64
import openai
import os
import traceback
from pathlib import Path
from app.config import Config
import uuid
from datetime import datetime

VOICE = [
    "alloy", "ash", "ballad", "coral", "echo",
    "fable", "onyx", "nova", "sage", "shimmer"
]
MODEL = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
RESPONSE_FORMAT = ["mp3", "opus", "aac", "flac", "wav", "pcm"]

text2speech_bp = Blueprint("text2speech", __name__)
client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)


@text2speech_bp.route("/text2speech", methods=["POST"])
@swag_from(os.path.join(os.path.dirname(__file__), "../../../docs/text2speech.yml"))
def audio_feedback_debug():
    try:
        model = request.form.get("model", "").strip()
        voice = request.form.get("voice", "").strip()
        input_text = request.form.get("input", "").strip()
        instructions = request.form.get("instructions", "").strip()
        format_ = request.form.get("response_format", "mp3").strip()
        speed = request.form.get("speed", "1.0").strip()

        if model not in MODEL:
            return jsonify({"success": False, "error": "Invalid or missing model"}), 400
        if voice not in VOICE:
            return jsonify({"success": False, "error": "Invalid or missing voice"}), 400
        if not input_text:
            return jsonify({"success": False, "error": "Missing input text"}), 400
        if format_ not in RESPONSE_FORMAT:
            return jsonify({"success": False, "error": "Invalid or unsupported response_format"}), 400

        try:
            speed = float(speed)
            if not 0.25 <= speed <= 4.0:
                raise ValueError
        except:
            return jsonify({"success": False, "error": "Speed must be between 0.25 and 4.0"}), 400

        audio_dir = Path(__file__).parent / "audio_result"
        audio_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"tts_{timestamp}.{format_}"
        speech_path = audio_dir / filename

        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=input_text,
            instructions = instructions,
            response_format=format_,
            speed=speed
        ) as response:
            response.stream_to_file(speech_path)

        return send_file(speech_path, mimetype=f"audio/{format_}", as_attachment=False)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
