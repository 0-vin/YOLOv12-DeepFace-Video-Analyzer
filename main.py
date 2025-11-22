import cv2
import os
import uuid
from ultralytics import YOLO
from deepface import DeepFace
from yt_dlp import YoutubeDL
import time
from typing import List, Tuple, Optional
import argparse
from scipy.spatial import distance


# ---------------- 유틸리티 함수들 (기존 동일) ----------------
def get_unique_name(base_name: str, extension: str) -> str:
    unique_id = uuid.uuid4().hex[:8]
    return f"{base_name}_{unique_id}.{extension}"


def merge_intervals(timestamps: List[float], gap: float = 2.0) -> List[Tuple[float, float]]:
    if not timestamps: return []
    timestamps.sort()
    intervals = []
    try:
        start = timestamps[0]
        prev = timestamps[0]
    except IndexError:
        return []
    for t in timestamps[1:]:
        if t - prev <= gap:
            prev = t
        else:
            intervals.append((start, prev))
            start = t
            prev = t
    intervals.append((start, prev))
    return intervals


def format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def download_video(url: str, output_dir: str = "downloads") -> Optional[str]:
    # (기존 다운로드 로직과 동일하여 내용 생략 없이 유지)
    os.makedirs(output_dir, exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    output_template = os.path.join(output_dir, f"%(title)s_{unique_id}.%(ext)s")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": True,  # 로그 너무 길어서 조금 줄임
        "postprocessors": [{'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'}],
    }
    print(f" 영상 다운로드 시작: {url}")
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            if info_dict:
                final_filepath = info_dict.get('filepath') or info_dict.get('_filename')
                # yt-dlp가 가끔 경로를 바로 안 줄 때가 있어서 안전장치
                if not final_filepath:
                    filename = ydl.prepare_filename(info_dict)
                    final_filepath = filename.replace('.webm', '.mp4').replace('.mkv', '.mp4')

                if final_filepath and os.path.exists(final_filepath):
                    print(f"✅ 다운로드 완료: {final_filepath}")
                    return final_filepath
            return None
    except Exception as e:
        print(f"❌ 다운로드 오류: {e}")
        return None


# ---------------- 영상 분석 클래스 ----------------
class VideoFaceAnalyzer:
    def __init__(self, yolo_model_path: str, deepface_model_name: str, device: str = 'cpu'):
        self.deepface_model_name = deepface_model_name  # 모델 이름 저장

        # 1. YOLO 모델 로드
        print(f"🚀 YOLO 모델 로드 중... ({yolo_model_path})")
        try:
            self.model = YOLO(yolo_model_path)
            self.model.to(device)
        except Exception as e:
            print(f"❌ YOLO 모델 로드 실패: {e}")
            raise

        # 2. DeepFace 모델 빌드 (첫 실행 딜레이 방지)
        print(f"🧠 DeepFace 모델 준비 중... ({self.deepface_model_name})")
        try:
            DeepFace.build_model(self.deepface_model_name)
        except Exception as e:
            print(f"⚠️ DeepFace 모델 빌드 중 경고 (무시 가능): {e}")

    def analyze_video(self, video_path: str, reference_face_path: str, checks_per_sec: float) -> List[float]:
        # 1. 참조 얼굴 임베딩 미리 계산
        print(f"📸 참조 얼굴 분석 중: {reference_face_path}")
        if not os.path.exists(reference_face_path):
            print("❌ 참조 얼굴 파일이 없습니다.")
            return []

        try:
            # SFace 사용 시 임계값을 조금 낮추는 것이 좋음 (ArcFace: 0.68, SFace: 0.4~0.5 권장)
            if self.deepface_model_name == "SFace":
                threshold = 0.5
            else:
                threshold = 0.68

                # 참조 이미지 임베딩 추출
            ref_results = DeepFace.represent(
                img_path=reference_face_path,
                model_name=self.deepface_model_name,  # ★ 설정한 모델 사용
                enforce_detection=False
            )
            ref_embedding = ref_results[0]["embedding"]
            print(f" 참조 얼굴 임베딩 완료 (Threshold: {threshold})")

        except Exception as e:
            print(f" 참조 얼굴 처리 오류: {e}")
            return []

        # 2. 비디오 설정
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30.0

        # 검사 간격 계산
        if checks_per_sec <= 0: checks_per_sec = 2.0
        skip_interval = int(fps / checks_per_sec)
        if skip_interval < 1: skip_interval = 1

        frame_idx = 0
        timestamps = []
        last_log_time = time.time()

        print(f"️ 분석 시작 (FPS: {fps:.2f} | {skip_interval}프레임마다 검사)")

        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_idx % skip_interval == 0:
                current_time = frame_idx / fps

                # 진행상황 로그
                if time.time() - last_log_time > 3.0:
                    print(f"   ... {format_time(current_time)} 진행 중")
                    last_log_time = time.time()

                # YOLO 감지
                results = self.model(frame, verbose=False, conf=0.5)  # confidence 0.5 이상만

                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 이미지 자르기 (경계선 처리 포함)
                    h, w = frame.shape[:2]
                    face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                    # 너무 작은 얼굴 무시 (속도 향상)
                    if face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
                        continue

                    try:
                        # DeepFace 입력 전 BGR -> RGB
                        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                        # 타겟 얼굴 임베딩 추출
                        target_res = DeepFace.represent(
                            img_path=face_rgb,
                            model_name=self.deepface_model_name,  # ★ 설정한 모델 사용
                            enforce_detection=False
                        )
                        target_embedding = target_res[0]["embedding"]

                        # 코사인 거리 계산
                        dist = distance.cosine(ref_embedding, target_embedding)

                        if dist <= threshold:
                            print(f" 찾음! {format_time(current_time)} (거리: {dist:.4f})")
                            timestamps.append(current_time)
                            break  # 한 프레임에서 찾으면 중복 검사 방지

                    except Exception:
                        continue

            frame_idx += 1

        cap.release()
        return timestamps


# ---------------- 메인 실행부 ----------------
def main():
    parser = argparse.ArgumentParser(description="영상 인물 탐지기")

    # [필수 옵션] 사용자가 꼭 입력해야 하는 것들
    parser.add_argument("--url", type=str, required=True, help="유튜브 영상 주소")
    parser.add_argument("--face", type=str, required=True, help="찾을 사람 이미지 파일 경로")
    parser.add_argument("--cps", type=float, required=True, help="초당 검사 횟수 (예: 2)")

    # [선택 옵션] 입력 안 하면 기본값 사용
    parser.add_argument("--yolo", type=str, default="/home/0vin/yolodeep/model_pt/yolov12n-face.pt",
                        help="YOLO 모델 경로 (.pt)")
    parser.add_argument("--deepface", type=str, default="ArcFace", help="DeepFace 모델 이름 (ArcFace, SFace 등)")
    parser.add_argument("--device", type=str, default="cpu", help="연산 장치 (cpu 또는 cuda)")

    args = parser.parse_args()

    # 실행
    video_file = download_video(args.url)
    if not video_file: return

    analyzer = VideoFaceAnalyzer(
        yolo_model_path=args.yolo,
        deepface_model_name=args.deepface,
        device=args.device
    )

    start_t = time.time()
    timestamps = analyzer.analyze_video(video_file, args.face, checks_per_sec=args.cps)
    end_t = time.time()

    # 결과 저장
    intervals = merge_intervals(timestamps)
    save_path = get_unique_name("result", "txt")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Video: {args.url}\n")
        f.write(f"Target: {args.face}\n")
        f.write(f"Model: YOLO={args.yolo} | DeepFace={args.deepface}\n")
        f.write(f"Time: {end_t - start_t:.2f}s\n")
        f.write("-" * 20 + "\n")
        for s, e in intervals:
            f.write(f"{format_time(s)} ~ {format_time(e)}\n")

    print(f"\n 결과가 저장되었습니다: {save_path}")


if __name__ == "__main__":
    main()