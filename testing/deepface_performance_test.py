import cv2
import time
import numpy as np
import os
from deepface import DeepFace

# --- [설정] 본인의 환경에 맞게 경로를 확인하세요 ---
# 1. 참조 얼굴 (등록된 얼굴)
REFERENCE_FACE_PATH = "/home/0vin/yolodeep/model_data/justin/justin_2.jpg"

# 2. 테스트할 크롭 이미지
# (주의: 확장자가 포함된 정확한 파일명이어야 합니다. 예: .jpg, .png)
TEST_CROP_PATH = "/home/0vin/yolodeep/model_data/justin/justin_1.DZcbMB-15GsG4aPgXKvE4AHaE8"

# 3. 테스트할 모델 목록 (FaceNet -> FaceNet512로 변경, SFace 추가)
# - ArcFace: 정확도 1위
# - FaceNet512: FaceNet의 개선 버전 (오류 해결 및 성능 향상)
# - SFace: 모바일/엣지 디바이스용 초경량 모델 (속도 기대주)
MODELS_TO_TEST = ["ArcFace", "VGG-Face", "SFace"]


# ------------------------------------------------

def test_deepface_performance():
    print("\n========== [DeepFace 성능 벤치마크 시작] ==========")

    # 1. 파일 존재 여부 확인
    if not os.path.exists(REFERENCE_FACE_PATH):
        print(f"오류: 참조 이미지가 없습니다 -> {REFERENCE_FACE_PATH}")
        return
    if not os.path.exists(TEST_CROP_PATH):
        print(f"오류: 테스트 이미지가 없습니다 -> {TEST_CROP_PATH}")
        return

    try:
        # 2. 이미지 로드 및 변환 (DeepFace는 RGB 포맷 권장)
        reference_img_bgr = cv2.imread(REFERENCE_FACE_PATH)
        test_crop_bgr = cv2.imread(TEST_CROP_PATH)

        if reference_img_bgr is None or test_crop_bgr is None:
            raise ValueError("이미지 로드 실패: 파일이 손상되었거나 지원하지 않는 형식입니다.")

        reference_img_rgb = cv2.cvtColor(reference_img_bgr, cv2.COLOR_BGR2RGB)
        test_crop_rgb = cv2.cvtColor(test_crop_bgr, cv2.COLOR_BGR2RGB)

        print(f">>> 이미지 로드 성공")
        print(f"    - Ref: {REFERENCE_FACE_PATH}")
        print(f"    - Test: {TEST_CROP_PATH}")

    except Exception as e:
        print(f"이미지 처리 중 치명적 오류: {e}")
        return

    results = {}

    # 3. 모델별 반복 테스트
    for model_name in MODELS_TO_TEST:
        print(f"\n----------------------------------------")
        print(f"★ 모델 테스트 중: {model_name}")
        print(f"----------------------------------------")

        try:
            # (1) 워밍업 (모델 가중치 다운로드/로드 및 초기화)
            # 최초 실행 시 가중치 파일(~/.deepface/weights/)을 다운로드하느라 오래 걸릴 수 있음
            print("  ... 모델 로드 및 예열(Warm-up) ...")
            _ = DeepFace.verify(
                img1_path=test_crop_rgb,
                img2_path=reference_img_rgb,
                model_name=model_name,
                enforce_detection=False,
                detector_backend='skip'  # 이미 잘린 얼굴이므로 탐지 생략
            )
            print("  ... 예열 완료! 측정 시작 ...")

            # (2) 속도 측정 (10회 반복 평균)
            timings = []
            verified_results = []
            distance_scores = []

            for i in range(10):
                start_time = time.time()

                res = DeepFace.verify(
                    img1_path=test_crop_rgb,
                    img2_path=reference_img_rgb,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend='skip'
                )

                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000

                timings.append(duration_ms)
                verified_results.append(res['verified'])
                distance_scores.append(res['distance'])

            avg_duration = np.mean(timings)
            avg_distance = np.mean(distance_scores)
            is_verified = verified_results[0]  # 첫 번째 결과 기준

            results[model_name] = {
                "time": avg_duration,
                "verified": is_verified,
                "distance": avg_distance
            }

            print(f"  ▶ 평균 소요 시간: {avg_duration:.2f} ms")
            print(f"  ▶ 검증 결과: {is_verified} (평균 거리값: {avg_distance:.4f})")

        except Exception as e:
            print(f"  !!! {model_name} 테스트 중 오류 발생: {e}")
            results[model_name] = {"time": -1.0, "verified": "Error", "distance": 0.0}

    # 4. 최종 결과 정렬 및 출력
    print("\n\n============= [최종 벤치마크 결과] =============")
    # 속도(Time)가 빠른 순서대로 정렬 (에러 난 것은 뒤로)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['time'] if x[1]['time'] > 0 else 99999)

    print(f"{'Model':<15} | {'Time (ms)':<15} | {'Verified':<10} | {'Distance'}")
    print("-" * 60)
    for model, data in sorted_results:
        if data['time'] != -1.0:
            # 검증 성공 여부에 따라 O / X 표시
            ver_mark = "O (True)" if data['verified'] else "X (False)"
            print(f"{model:<15} | {data['time']:<15.2f} | {ver_mark:<10} | {data['distance']:.4f}")
        else:
            print(f"{model:<15} | {'Error':<15} | {'Error':<10} | -")
    print("================================================")
    print("* Distance가 낮을수록 얼굴이 더 비슷하다는 뜻입니다.")
    print("* SFace는 속도가 빠르지만 Distance 임계값이 다를 수 있습니다.")


if __name__ == "__main__":
    test_deepface_performance()