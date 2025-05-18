import os
import sys
import argparse

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import threading
from flask import Flask, Response
from src.services.detection_service import DetectionService

app = Flask(__name__)
last_frame = None  # Variável global para armazenar o último frame processado

# Configuração do parser de argumentos
parser = argparse.ArgumentParser(description='GatekeeperX - Sistema de detecção inteligente')
parser.add_argument('--camera-ip', type=str, help='IP da câmera (ex: 192.168.0.100)')
parser.add_argument('--username', type=str, default='Dannark', help='Usuário da câmera (padrão: Dannark)')
parser.add_argument('--password', type=str, default='23021994', help='Senha da câmera (padrão: 23021994)')
args = parser.parse_args()

# IP padrão da câmera
DEFAULT_CAMERA_IP = "192.168.0.100"

def build_rtsp_url(ip, username=None, password=None):
    """Constrói a URL RTSP a partir do IP e credenciais"""
    if username and password:
        return f"rtsp://{username}:{password}@{ip}:554/stream1"
    return f"rtsp://{ip}:554/stream1"

def processing_loop():
    global last_frame
    # Usa o IP fornecido via argumento ou o IP padrão
    camera_ip = args.camera_ip if args.camera_ip else DEFAULT_CAMERA_IP
    rtsp_url = build_rtsp_url(camera_ip, args.username, args.password)
    print(f"Conectando à câmera em: {rtsp_url}")
    
    try:
        detection_service = DetectionService(camera_ip=rtsp_url)
        detection_service.start()
        while True:
            frame, results, now = detection_service.process_frame()
            if frame is None:
                print("Erro ao processar frame. Tentando reconectar...")
                break
            annotated = detection_service.draw_annotations(frame, results, now)
            last_frame = annotated  # Atualiza o frame global
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Erro ao conectar com a câmera: {str(e)}")
        print("Verifique se:")
        print("1. O IP está correto")
        print("2. A câmera está ligada e conectada à rede")
        print("3. A porta 554 está aberta")
        print("4. O usuário e senha da câmera estão corretos (se necessário)")
    finally:
        if 'detection_service' in locals():
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
