import time
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine

# --- [설정] ---
IMG_PATH = "/home/0vin/yolodeep/model_data/justin/justin_1.DZcbMB-15GsG4aPgXKvE4AHaE8"
MODEL_NAME = "ArcFace"  # 또는 SFace, VGG-Face 등 측정하고 싶은 모델


def measure_split_performance():
    print(f"[{MODEL_NAME}] 분리 성능 측정 시작...\n")

    # 1. 이미지 로드 (측정 제외)
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. 워밍업 (첫 실행 속도 제외)
    print(">>> 워밍업 중 (Warm-up)...")
    _ = DeepFace.represent(img_path=img, model_name=MODEL_NAME, enforce_detection=False)

    # 임시 벡터 생성 (거리 계산 측정용)
    vec1 = [0.1] * 512  # 임의의 벡터
    vec2 = [0.1] * 512
    if MODEL_NAME == "VGG-Face":  # VGG는 차원수가 다름
        vec1 = [0.1] * 2622
        vec2 = [0.1] * 2622

    # ==========================================
    # [측정 1] 임베딩 추출 시간 (DeepFace.represent)
    # ==========================================
    print(">>> 1. 임베딩 추출 시간 측정 중...")
    start_time = time.time()

    # 실제 얼굴을 벡터(숫자 리스트)로 바꾸는 작업
    embedding_objs = DeepFace.represent(
        img_path=img,
        model_name=MODEL_NAME,
        enforce_detection=False
    )
    embedding_vector = embedding_objs[0]["embedding"]

    end_time = time.time()
    represent_time_ms = (end_time - start_time) * 1000

    # ==========================================
    # [측정 2] 거리 계산 시간 (Cosine Similarity)
    # ==========================================
    print(">>> 2. 거리 계산 시간 측정 중...")

    # 추출된 벡터끼리 단순 수학 계산하는 작업
    vec_a = embedding_vector
    vec_b = embedding_vector  # 테스트니 자기 자신과 비교

    start_time = time.time()

    _ = cosine(vec_a, vec_b)

    end_time = time.time()
    calc_time_ms = (end_time - start_time) * 1000

    # ==========================================
    # [결과 출력] 보고서 빈칸용 데이터
    # ==========================================
    print(f"\n========== [{MODEL_NAME} 결과] ==========")
    print(f"1. 임베딩 추출 (DeepFace.represent): {represent_time_ms:.4f} ms")
    print(f"2. 거리 비교 (Cosine Similarity)   : {calc_time_ms:.6f} ms")
    print("=======================================")
    print(f"결론: 이미 추출된 벡터만 있다면, 속도가 약 {represent_time_ms / calc_time_ms:.0f}배 빨라짐")


if __name__ == "__main__":
    measure_split_performance()