import cv2
import time
from deepface import DeepFace
import numpy as np
import os
from scipy.spatial import distance  # 거리 계산을 위해 추가

# --- 테스트용 이미지 준비 ---
REF_FACE_PATH = "/home/0vin/yolodeep/model_data/justin/justin_2.jpg"
TEST_CROP_PATH = "/home/0vin/yolodeep/model_data/justin/justin_1.DZcbMB-15GsG4aPgXKvE4AHaE8"


def create_dummy_image(path):
    if not os.path.exists(path):
        print(f"경고: {path} 가짜 이미지 생성.")
        dummy_img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        cv2.imwrite(path, dummy_img)


create_dummy_image(REF_FACE_PATH)
create_dummy_image(TEST_CROP_PATH)

MODEL_NAME = "ArcFace"
METRIC = "cosine"
ITERATIONS = 10

print("--- DeepFace 캐시 성능 테스트 시작 ---")

# --- 1. "No-Cache" 방식 (기존과 동일) ---
try:
    print("\n--- 1. 'No-Cache' 방식 (매번 verify 호출) ---")

    # Warm-up
    DeepFace.verify(TEST_CROP_PATH, REF_FACE_PATH, model_name=MODEL_NAME, enforce_detection=False)

    start_total = time.time()
    for i in range(ITERATIONS):
        # 매번 두 이미지를 모델에 통과시킴 (무거운 작업 x 2)
        DeepFace.verify(TEST_CROP_PATH, REF_FACE_PATH,
                        model_name=MODEL_NAME, distance_metric=METRIC,
                        enforce_detection=False)
    end_total = time.time()

    avg_no_cache_time = ((end_total - start_total) / ITERATIONS) * 1000
    print(f"★ 'No-Cache' 평균 시간: {avg_no_cache_time:.2f} ms")

except Exception as e:
    print(f"'No-Cache' 오류: {e}")
    avg_no_cache_time = -1.0

# --- 2. "Cache" 방식 (임베딩 미리 계산 + 수동 거리 측정) ---
try:
    print("\n--- 2. 'Cache' 방식 (임베딩 미리 계산 후 비교) ---")

    # (A) 참조 얼굴 임베딩 미리 계산 (루프 밖에서 1회 수행)
    # verify 대신 represent를 사용하여 벡터값만 추출합니다.
    print("... 참조 얼굴 임베딩 계산 중 ...")
    ref_results = DeepFace.represent(REF_FACE_PATH, model_name=MODEL_NAME, enforce_detection=False)
    ref_embedding = ref_results[0]["embedding"]  # 벡터 리스트 추출

    # 임계값 가져오기 (ArcFace + Cosine 기준)
    threshold = 0.68  # ArcFace 기본값 (모델/메트릭에 따라 다름)

    # Warm-up (테스트 이미지 임베딩 추출)
    DeepFace.represent(TEST_CROP_PATH, model_name=MODEL_NAME, enforce_detection=False)

    start_cache = time.time()
    for i in range(ITERATIONS):
        # 1. 테스트 이미지의 임베딩만 계산 (무거운 작업 x 1)
        target_results = DeepFace.represent(TEST_CROP_PATH, model_name=MODEL_NAME, enforce_detection=False)
        target_embedding = target_results[0]["embedding"]

        # 2. 수학적 거리 계산 (아주 가벼운 연산)
        if METRIC == "cosine":
            dist = distance.cosine(ref_embedding, target_embedding)
        elif METRIC == "euclidean":
            dist = distance.euclidean(ref_embedding, target_embedding)

        # 3. 판별 (Verify 로직 수동 구현)
        is_match = dist <= threshold

    end_cache = time.time()

    avg_cache_time = ((end_cache - start_cache) / ITERATIONS) * 1000
    print(f"★ 'Cache' 평균 시간: {avg_cache_time:.2f} ms")

except Exception as e:
    print(f"'Cache' 오류: {e}")
    avg_cache_time = -1.0

# --- 3. 최종 결과 ---
print("\n--- 최종 요약 ---")
print(f"No-Cache: {avg_no_cache_time:.2f} ms")
print(f"Cache   : {avg_cache_time:.2f} ms")

if avg_cache_time > 0 and avg_no_cache_time > 0:
    speedup = avg_no_cache_time / avg_cache_time
    print(f"▶ 성능 향상: {speedup:.1f} 배 더 빠름")