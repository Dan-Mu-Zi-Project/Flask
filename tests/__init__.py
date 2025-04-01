import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace

FACE_DETECTION_MODEL = "retinaface"
FACE_RECOGNITION_MODEL = "ArcFace"


def load_image_from_path(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다.")
    return img


def show_face_visualization(original_img, face_obj):
    """
    원본 이미지와 추출된 얼굴을 시각적으로 보여줌
    """
    face_rgb = face_obj["face"]
    facial_area = face_obj["facial_area"]

    annotated_img = original_img.copy()
    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    if face_rgb.dtype != np.uint8:
        face_rgb = (face_rgb * 255).astype(np.uint8)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original with Box")
    plt.imshow(annotated_img_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Cropped Face")
    plt.imshow(face_rgb)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def test_single_face_embedding_with_visual(image_path):
    img = load_image_from_path(image_path)

    print("[*] 얼굴 추출 중...")
    faces = DeepFace.extract_faces(
        img_path=img,
        detector_backend=FACE_DETECTION_MODEL,
        enforce_detection=True,
        align=True,
        expand_percentage=0,
        grayscale=False,
        color_face="rgb",
        normalize_face=True
    )

    if not faces:
        print("[!] 얼굴이 감지되지 않았습니다.")
        return

    face_obj = faces[0]
    print(f"[*] 얼굴 영역: {face_obj['facial_area']}")

    show_face_visualization(img, face_obj)

    try:
        face_rgb = face_obj["face"]
        if face_rgb.dtype != np.uint8:
            face_rgb = (face_rgb * 255).astype(np.uint8)

        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

        result = DeepFace.represent(
            img_path=face_bgr,
            model_name=FACE_RECOGNITION_MODEL,
            detector_backend="skip",
            enforce_detection=False
        )

        print("[✅] 임베딩 추출 성공!")
        print(f"[*] 벡터 길이: {len(result[0]['embedding'])}")

    except Exception as e:
        print(f"[❌] 임베딩 추출 실패: {e}")


# 실행
if __name__ == "__main__":
    test_image_path = r"C:\Flask\tests\dynamicduo.jpg"  # 테스트 이미지 경로
    test_single_face_embedding_with_visual(test_image_path)
