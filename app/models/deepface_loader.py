from app.logger_config import setup_logger
from deepface import DeepFace
import numpy as np
import logging

logger = setup_logger(__name__, logging.DEBUG)

FACE_RECONGITION_MODEL = "Facenet512"
FACE_DETECTION_MODEL = "yolov11n"

logger.debug("🧠 DeepFace 모델 로딩 중...")
preload_recongition_model = DeepFace.build_model(FACE_RECONGITION_MODEL)

dummy_image = np.zeros((160, 160, 3), dtype=np.uint8)
DeepFace.represent(
    img_path=dummy_image,
    model_name=FACE_RECONGITION_MODEL,
    detector_backend=FACE_DETECTION_MODEL,
    enforce_detection=False
)

logger.debug("✅ 모델 로딩 완료")