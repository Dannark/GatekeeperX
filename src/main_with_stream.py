import cv2
import threading
from flask import Flask, Response
from src.services.detection_service import DetectionService
import os

app = Flask(__name__)
last_frame = None  # Variável global para armazenar o último frame processado

def processing_loop():
    global last_frame
    detection_service = DetectionService()
    try:
        while True:
            frame, results, now = detection_service.process_frame()
            if frame is None:
                break
            annotated = detection_service.draw_annotations(frame, results, now)
            last_frame = annotated  # Atualiza o frame global
            # cv2.imshow("GatekeeperX", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        detection_service.cleanup()

def gen_frames():
    global last_frame
    while True:
        if last_frame is not None:
            ret, buffer = cv2.imencode('.jpg', last_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Stream de vídeo</title>
        <style>
            body { background: #222; color: #fff; text-align: center; }
            .video-container {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            img {
                width: 100%;
                max-width: 900px;
                height: auto;
                border: 4px solid #444;
                border-radius: 12px;
                background: #000;
                box-shadow: 0 0 24px #000a;
            }
        </style>
    </head>
    <body>
        <h1>Stream de vídeo</h1>
        <div class="video-container">
            <img src='/video_feed' alt='Stream de vídeo'>
        </div>
    </body>
    </html>
    """

@app.route('/logs')
def logs():
    log_path = 'gatekeeperx.log'
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            content = f.read()
        return f"<pre style='background:#111;color:#0f0;padding:16px;'>{content}</pre>"
    else:
        return "<pre style='color:red'>Nenhum log encontrado.</pre>" 

if __name__ == "__main__":
    t = threading.Thread(target=processing_loop)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5050, debug=False)
