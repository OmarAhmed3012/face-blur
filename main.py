from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
import asyncio
import cv2
import numpy as np
import tempfile
import os
import subprocess
import shutil
import mediapipe as mp
from concurrent.futures import ProcessPoolExecutor

app = FastAPI(title="Video-Blur & Voice-Change Service with Parallel Frame Processing")

# Initialize MediaPipe face detector once
dp = mp.solutions.face_detection
detector = dp.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Locate FFmpeg (fallback if needed)
FALLBACK_FFMPEG = r"C:\Users\User\AppData\Local\Microsoft\WinGet\Links\ffmpeg.EXE"
FFMPEG = shutil.which("ffmpeg") or (FALLBACK_FFMPEG if os.path.isfile(FALLBACK_FFMPEG) else None)
if not FFMPEG:
    raise RuntimeError("FFmpeg not found. Please install and ensure it's on PATH.")


def heavy_process(input_path: str, blur_all: bool, change_voice: bool) -> str:
    # Create temp working directory
    workdir = tempfile.mkdtemp()
    base = os.path.basename(input_path)

    # Read all frames into memory
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frames = []
    cap = cv2.VideoCapture(input_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Define per-frame work function
    def process_frame(frame: np.ndarray) -> bytes:
        # Apply blur logic
        if blur_all:
            k = max(w, h) // 10
            if k % 2 == 0:
                k += 1
            processed = cv2.blur(frame, (k, k))
        else:
            # Downscale for faster face detection
            small = cv2.resize(frame, (w // 2, h // 2))
            results = detector.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            mask = np.zeros((h, w), dtype=np.uint8)
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    w_box = int(bbox.width * w)
                    h_box = int(bbox.height * h)
                    pad_w = int(w_box * 0.6)
                    pad_h = int(h_box * 0.3)
                    x0 = max(0, x1 - pad_w)
                    y0 = max(0, y1 - pad_h)
                    x2 = min(w, x1 + w_box + pad_w)
                    y2 = min(h, y1 + h_box + pad_h)
                    mask[y0:y2, x0:x2] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 81))
            mask = cv2.dilate(mask, kernel, iterations=1)
            k = max(w, h) // 10
            if k % 2 == 0:
                k += 1
            blurred = cv2.blur(frame, (k, k))
            processed = frame.copy()
            processed[mask == 255] = blurred[mask == 255]
        # Rotate 90° CCW
        rotated = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return rotated.tobytes()

    # Parallel processing across CPU cores
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        processed_bytes = list(executor.map(process_frame, frames))

    # Setup FFmpeg encoding pipeline for video-only
    out_w, out_h = h, w  # after rotation
    video_only = os.path.join(workdir, f"video_only_{base}")
    cmd = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}",
        "-r", str(int(fps)),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        video_only
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame_bytes in processed_bytes:
        proc.stdin.write(frame_bytes)
    proc.stdin.close()
    proc.wait()

    # Step 2: Audio processing
    audio_only = os.path.join(workdir, f"audio_{base}")
    if change_voice:
        filter_str = "asetrate=44100*0.9,aresample=44100,atempo=1.111"
        subprocess.run([
            FFMPEG, "-y", "-i", input_path,
            "-vn", "-af", filter_str,
            "-c:a", "aac", audio_only
        ], check=True)
    else:
        subprocess.run([
            FFMPEG, "-y", "-i", input_path,
            "-vn", "-c:a", "copy", audio_only
        ], check=True)

    # Step 3: Mux video and audio streams
    final_out = os.path.join(workdir, f"final_{base}")
    subprocess.run([
        FFMPEG, "-y",
        "-i", video_only,
        "-i", audio_only,
        "-c:v", "copy", "-c:a", "aac", final_out
    ], check=True)

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
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    tmp.write(await file.read())
    tmp.flush()
    tmp.close()

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, heavy_process, tmp.name, blur_all, change_voice)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(result, media_type="video/mp4", filename=os.path.basename(result))
