from ultralytics import YOLO
import cv2
from datetime import datetime, timedelta

# Carrega modelo YOLOv8 (baixado automaticamente)
model = YOLO("yolov8n.pt")  # Você pode trocar por yolov8s.pt para mais precisão

rtsp_url = "rtsp://Dannark:23021994@192.168.0.104:554/stream1"
cap = cv2.VideoCapture(rtsp_url)

active_objects = {}
TIMEOUT_SECONDS = 3  # tolerância para considerar que saiu da cena

LOG_LEVEL = 1  # 0 = silencioso, 1 = normal, 2 = somente alertas
def log(level, message):
    if level >= LOG_LEVEL:
        print(message)

# Área de interesse (percentual da largura e altura)
AREA_X_MIN = 0.1
AREA_X_MAX = 0.95
AREA_Y_MIN = 0.3
AREA_Y_MAX = 0.95

while True:
    ret, frame = cap.read()
    if not ret:
        log(1, "Erro ao acessar o stream")
        break

    # Faz rastreamento com o modelo
    results = model.track(source=frame, persist=True, conf=0.5, verbose=False)

    now = datetime.now()
    current_ids = set()

    for r in results:
        for box in r.boxes:
            obj_id = int(box.id[0]) if box.id is not None else None
            if obj_id is None:
                continue
            current_ids.add(obj_id)

            x1, y1, x2, y2 = box.xyxy[0]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            inside_area = (
                AREA_X_MIN * frame.shape[1] < cx < AREA_X_MAX * frame.shape[1] and
                AREA_Y_MIN * frame.shape[0] < cy < AREA_Y_MAX * frame.shape[0]
            )

            if obj_id not in active_objects:
                active_objects[obj_id] = {
                    "label": model.names[int(box.cls[0])],
                    "last_seen": now,
                    "entry_time": now,
                    "logged_exit": False,
                    "alerted_level": 0,
                    "entered_area": False
                }
                label = active_objects[obj_id]["label"]
                log(1, f"ID: {obj_id} - {label} ENTROU às {now.strftime('%H:%M:%S')}")
            else:
                active_objects[obj_id]["last_seen"] = now
                active_objects[obj_id]["logged_exit"] = False
                if inside_area:
                    active_objects[obj_id]["entered_area"] = True

    for oid, data in list(active_objects.items()):
        if now - data["last_seen"] > timedelta(seconds=TIMEOUT_SECONDS) and not data["logged_exit"]:
            log(1, f"ID: {oid} - {data['label']} SAIU às {now.strftime('%H:%M:%S')}")
            active_objects[oid]["logged_exit"] = True

    for oid, data in active_objects.items():
        duration = (now - data["entry_time"]).total_seconds()
        if data["entered_area"] and duration > 10 and data["alerted_level"] < 2:
            log(2, f"AVISO NÍVEL 2: ID {oid} - {data['label']} está há {int(duration)}s na área! ({now.strftime('%H:%M:%S')})")
            data["alerted_level"] = 2

    # Exibe resultado com anotações
    annotated = results[0].plot()
    # Desenhar área de interesse (30% a 70% da largura e altura)
    h, w, _ = frame.shape
    x1, y1 = int(AREA_X_MIN * w), int(AREA_Y_MIN * h)
    x2, y2 = int(AREA_X_MAX * w), int(AREA_Y_MAX * h)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow("Detecção - YOLOv8", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()