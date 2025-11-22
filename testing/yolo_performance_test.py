import cv2
import time
import os
import argparse  # 명령행 인자 파싱을 위해 추가
from ultralytics import YOLO


def test_yolo_performance():
    # --- 1. 명령행 인자 설정 (argparse) ---
    parser = argparse.ArgumentParser(description='YOLO Performance Test')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model (.pt)')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size (default: 640)')

    args = parser.parse_args()

    # 변수 할당
    YOLO_MODEL_PATH = args.model
    VIDEO_PATH = args.video
    img_size = args.imgsz

    TOTAL_FRAMES_TO_TEST = 300
    DEVICE = "cpu"

    print(f"--- YOLO 모델 로드 중: {YOLO_MODEL_PATH} ---")
    print(f"--- 추론 이미지 크기(imgsz): {img_size} ---")

    try:
        model = YOLO(YOLO_MODEL_PATH)
        # model.to(DEVICE) # Ultralytics는 자동으로 장치를 잡지만 명시해도 됨
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {VIDEO_PATH}")
        return

    print(f"--- 비디오 파일 열기 성공: {VIDEO_PATH} ---")
    print(f"--- 지금 다른 터미널에서 'htop'을 실행하여 CPU/MEM을 확인하세요 ---")
    print(f"--- 5초 후 {TOTAL_FRAMES_TO_TEST} 프레임 분석을 시작합니다 ---")
    time.sleep(5)

    frame_count = 0
    start_time = time.time()

    while frame_count < TOTAL_FRAMES_TO_TEST:
        ret, frame = cap.read()
        if not ret:
            print("비디오 끝에 도달했습니다.")
            break

        # ★★★ YOLO 추론 실행 ★★★
        try:
            # cv2.resize를 직접 하지 않고, YOLO에게 imgsz 인자로 넘기는 것이 훨씬 효율적입니다.
            # YOLO 내부적으로 리사이징과 패딩(Letterbox)을 최적화해서 처리합니다.
            results = model(frame, imgsz=img_size, verbose=False, device=DEVICE)

            frame_count += 1

            if frame_count % 30 == 0:
                print(f"  ... {frame_count} 프레임 처리 중 ...")

        except Exception as e:
            print(f"YOLO 추론 중 오류: {e}")
            break

    end_time = time.time()
    cap.release()

    total_time = end_time - start_time

    if total_time > 0:
        fps = frame_count / total_time
    else:
        fps = float('inf')

    print("\n--- 분석 완료 ---")
    print(f"모델: {os.path.basename(YOLO_MODEL_PATH)}")
    print(f"설정된 imgsz: {img_size}")
    print(f"총 분석 시간: {total_time:.2f} 초")
    print(f"총 처리 프레임: {frame_count} 개")
    print(f"★★★ 평균 FPS: {fps:.2f} ★★★")


if __name__ == "__main__":
    test_yolo_performance()