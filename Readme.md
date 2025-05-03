# Video Blur & Voice Change Service

A FastAPI service to blur faces in video, rotate frames, and optionally deepen the voice. Uses MediaPipe for face detection, OpenCV for blurring, and FFmpeg for encoding.

## Project Structure

```
video-blur-service/
├── Dockerfile
├── docker-compose.yml
├── main.py
├── requirements.txt
├── models/
│   ├── deploy.prototxt.txt
│   └── res10_300x300_ssd_iter_140000_fp16.caffemodel
├── README.md
└── (temporary output dirs created at runtime)
```

## Dependencies

* Python 3.10
* FFmpeg on PATH
* Python packages listed in `requirements.txt`

## Running Locally

1. **Activate your venv** (Python 3.10):

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate      # Linux/macOS
   .\venv\Scripts\Activate.ps1 # Windows PowerShell
   ```

2. **Install Python dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install `uvloop` (optional, Linux/macOS only)**:

   ```bash
   pip install uvloop
   ```

4. **Launch the server with 2 workers**:

   ```bash
   # On Linux/macOS (with uvloop):
   uvicorn main:app --workers 2 --loop uvloop --host 0.0.0.0 --port 8000

   # On Windows (omit uvloop):
   uvicorn main:app --workers 2 --host 0.0.0.0 --port 8000
   ```

## Adjusting FFmpeg Encoding

To trade off quality vs. speed/size, open `main.py` and locate the FFmpeg command in `heavy_process()`. Modify:

```diff
-   "-preset", "fast",
-   "-crf",    "23",
+   "-preset", "ultrafast",  # faster encoding, larger output
+   "-crf",    "28",         # higher CRF: more compression, lower quality
```

Restart the server after changes.

## Testing Endpoints

* **Health check**: `GET /health` → `{ "status": "ok" }`
* **Blur & optional voice change**:

  ```bash
  curl -X POST "http://localhost:8000/blur-video?blur_all=false&change_voice=true" \
       -F "file=@/path/to/video.mp4" \
       --output result.mp4
  ```

## Docker

Build and run using Dockerfile:

```bash
docker build -t video-blur-service .
docker run --rm -p 8000:8000 video-blur-service
```

Or with Docker Compose:

```bash
docker-compose up --build
```
