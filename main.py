import os
import logging
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
import asyncio
import cv2
import numpy as np
import tempfile
import subprocess
import shutil
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor

# Suppress verbose TensorFlow/MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Suppress Mediapipe/C++ warning logs (glog)
os.environ['GLOG_minloglevel'] = '2'
os.environ['GLOG_logtostderr'] = '1'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
# Lower MediaPipe logger level
mp_logger = logging.getLogger('mediapipe')
mp_logger.setLevel(logging.ERROR)

# Set up application logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("video_blur")

app = FastAPI(title="Video-Blur & Voice-Change Service with 2-Core Parallelism")

# Initialize MediaPipe face detection config
dp = mp.solutions.face_detection

# Locate FFmpeg (fallback if needed)
FALLBACK_FFMPEG = r"C:\Users\User\AppData\Local\Microsoft\WinGet\Links\ffmpeg.EXE"
FFMPEG = shutil.which("ffmpeg") or (FALLBACK_FFMPEG if os.path.isfile(FALLBACK_FFMPEG) else None)
if not FFMPEG:
    logger.error("FFmpeg not found. Please install and ensure it's on PATH.")
    raise RuntimeError("FFmpeg not found. Please install and ensure it's on PATH.")


def process_frame(args):
    frame, w, h, blur_all = args
    with dp.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        if blur_all:
            k = max(w, h) // 10
            if k % 2 == 0:
                k += 1
            processed = cv2.blur(frame, (k, k))
        else:
            small = cv2.resize(frame, (w // 2, h // 2))
            results = detector.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            mask = np.zeros((h, w), dtype=np.uint8)
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w); y1 = int(bbox.ymin * h)
                    w_box = int(bbox.width * w); h_box = int(bbox.height * h)
                    pad_w = int(w_box * 0.6); pad_h = int(h_box * 0.3)
                    x0 = max(0, x1 - pad_w); y0 = max(0, y1 - pad_h)
                    x2 = min(w, x1 + w_box + pad_w); y2 = min(h, y1 + h_box + pad_h)
                    mask[y0:y2, x0:x2] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 81))
            mask = cv2.dilate(mask, kernel, iterations=1)
            k = max(w, h) // 10
            if k % 2 == 0: k += 1
            blurred = cv2.blur(frame, (k, k))
            processed = frame.copy()
            processed[mask == 255] = blurred[mask == 255]
    rotated = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated.tobytes()


def heavy_process(input_path: str, blur_all: bool, change_voice: bool) -> str:
    logger.info("=== /blur-video START (blur_all=%s, change_voice=%s) ===", blur_all, change_voice)
    workdir = tempfile.mkdtemp()
    base = os.path.basename(input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("Cannot open video file: %s", input_path)
        raise RuntimeError("Cannot open video file")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # --- VIDEO METADATA ---
    # Total frames and duration
    temp_cap = cv2.VideoCapture(input_path)
    frame_count = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    temp_cap.release()
    duration = frame_count / fps if fps else 0
    logger.info("===== VIDEO METADATA =====")
    logger.info("File: %s", base)
    logger.info("Resolution: %dx%d", w, h)
    logger.info("FPS: %.2f", fps)
    logger.info("Total frames: %d", frame_count)
    logger.info("Duration: %.2f seconds", duration)
    logger.info("===== END METADATA =====")
    logger.info("Video %s: %dx%d @ %.2f FPS", base, w, h, fps)

    frames = []
    cap = cv2.VideoCapture(input_path)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    total_frames = len(frames)
    logger.info("Loaded %d frames into memory", total_frames)

    args_list = [(f, w, h, blur_all) for f in frames]
    logger.info("Processing frames in parallel on 2 threads")
    with ThreadPoolExecutor(max_workers=2) as exec:
        processed_bytes = list(exec.map(process_frame, args_list))
    logger.info("Frame processing done")

    out_w, out_h = h, w
    video_only = os.path.join(workdir, f"video_only_{base}")
    cmd = [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{out_w}x{out_h}", "-r", str(int(fps)), "-i", "pipe:0",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
           "-pix_fmt", "yuv420p", video_only]
    logger.info("Encoding video with FFmpeg to %s", video_only)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fb in processed_bytes:
        proc.stdin.write(fb)
    proc.stdin.close(); proc.wait()
    logger.info("Video encoding complete")

    audio_only = os.path.join(workdir, f"audio_{base}")
    if change_voice:
        logger.info("Applying voice change filter")
        filter_str = "asetrate=44100*0.9,aresample=44100,atempo=1.111"
        subprocess.run([FFMPEG, "-y", "-i", input_path,
                        "-vn", "-af", filter_str,
                        "-c:a", "aac", audio_only], check=True)
    else:
        logger.info("Copying original audio stream")
        subprocess.run([FFMPEG, "-y", "-i", input_path,
                        "-vn", "-c:a", "copy", audio_only], check=True)

    final_out = os.path.join(workdir, f"final_{base}")
    logger.info("Muxing video + audio to %s", final_out)
    subprocess.run([FFMPEG, "-y", "-i", video_only,
                    "-i", audio_only,
                    "-c:v", "copy", "-c:a", "aac", final_out], check=True)
    logger.info("=== /blur-video COMPLETE, output: %s ===", final_out)
    return final_out


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/blur-video")
async def blur_video(
    file: UploadFile = File(...),
    blur_all: bool = Query(False, description="Blur entire frame"),
    change_voice: bool = Query(False, description="Apply voice transformation")
):
    logger.info("Endpoint /blur-video called: blur_all=%s, change_voice=%s", blur_all, change_voice)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    tmp.write(await file.read()); tmp.flush(); tmp.close()
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, heavy_process, tmp.name, blur_all, change_voice)
    except Exception as e:
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail=str(e))
    logger.info("Returning result file %s", result)
    return FileResponse(result, media_type="video/mp4", filename=os.path.basename(result))
