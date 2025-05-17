from ultralytics import YOLO
import cv2
from datetime import datetime

# Carrega modelo YOLOv8 (baixado automaticamente)
model = YOLO("yolov8n.pt")  # Você pode trocar por yolov8s.pt para mais precisão

rtsp_url = "rtsp://Dannark:23021994@192.168.0.104:554/stream1"
cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar o stream")
        break

    # Faz predição com o modelo
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # Processa detecções
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            print(f"Detectado: {label} com {conf*100:.2f}% às {datetime.now().strftime('%H:%M:%S')}")

    # Exibe resultado com anotações
    annotated = results[0].plot()
    cv2.imshow("Detecção - YOLOv8", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()