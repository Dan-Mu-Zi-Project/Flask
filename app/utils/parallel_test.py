import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from deepface import DeepFace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_list = [
    os.path.join(BASE_DIR, "..", "..", "tests", "beenzino.jpeg"),
    os.path.join(BASE_DIR, "..", "..", "tests", "choija.png"),
    os.path.join(BASE_DIR, "..", "..", "tests", "choijunghoon.jpg"),
    os.path.join(BASE_DIR, "..", "..", "tests", "choiyuree.jpg"),
    os.path.join(BASE_DIR, "..", "..", "tests", "gaeko.jpg"),
]

FACE_RECONGITION_MODEL = "Facenet512"
FACE_DETECTION_MODEL = "yolov11n"


def set_tf_env(omp="1", intra="1", inter="1"):
    os.environ["OMP_NUM_THREADS"] = omp
    os.environ["TF_NUM_INTRAOP_THREADS"] = intra
    os.environ["TF_NUM_INTEROP_THREADS"] = inter


def extract(img_path):
    return DeepFace.represent(
        img_path=img_path,
        # model_name=FACE_RECONGITION_MODEL,
        # detector_backend=FACE_DETECTION_MODEL,
        enforce_detection=False,
    )


log_output = []


def run_executor(name, executor_class, max_workers, omp="1", intra="1", inter="1"):
    set_tf_env(omp, intra, inter)
    start = time.time()
    results = []

    local_output = []

    try:
        with executor_class(max_workers=max_workers) as executor:
            futures = [executor.submit(extract, img) for img in img_list]
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    local_output.append(f"[❌ 오류] {e}")

        elapsed = time.time() - start
        local_output.append(
            f"\n=== 🚀 {name} | workers={max_workers} | OMP={omp} INTRA={intra} INTER={inter} ==="
        )
        local_output.append(
            f"[✅ 결과] 처리 시간: {elapsed:.2f}초 | 성공: {len(results)}/{len(img_list)}"
        )
    except Exception as e:
        local_output.append(f"[🔥 실패] {name}: {e}")

    log_output.extend(local_output)


test_cases = [
    {
        "name": "ThreadPool-2w",
        "executor": ThreadPoolExecutor,
        "workers": 2,
        "omp": "1",
        "intra": "1",
        "inter": "1",
    },
    {
        "name": "ThreadPool-4w",
        "executor": ThreadPoolExecutor,
        "workers": 4,
        "omp": "1",
        "intra": "1",
        "inter": "1",
    },
    {
        "name": "ProcessPool-2w",
        "executor": ProcessPoolExecutor,
        "workers": 2,
        "omp": "1",
        "intra": "1",
        "inter": "1",
    },
    {
        "name": "ProcessPool-4w",
        "executor": ProcessPoolExecutor,
        "workers": 4,
        "omp": "1",
        "intra": "1",
        "inter": "1",
    },
]


def run_selected_tests(cases):
    for case in cases:
        run_executor(
            name=case["name"],
            executor_class=case["executor"],
            max_workers=case["workers"],
            omp=case["omp"],
            intra=case["intra"],
            inter=case["inter"],
        )
    for output in log_output:
        print(output)


if __name__ == "__main__":
    run_selected_tests(test_cases)
