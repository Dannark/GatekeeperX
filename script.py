from ultralytics import YOLO
import cv2
from datetime import datetime, timedelta

# Carrega modelo YOLOv8 (baixado automaticamente)
model = YOLO("yolov8n.pt")  # Você pode trocar por yolov8s.pt para mais precisão

rtsp_url = "rtsp://Dannark:23021994@192.168.0.104:554/stream1"
cap = cv2.VideoCapture(rtsp_url)

active_objects = {}
TIMEOUT_SECONDS = 3  # tolerância para considerar que saiu da cena
AREA_TIMEOUT_SECONDS = 3  # tolerância para considerar que saiu da área
AREA_PRESENCE_THRESHOLD = 10  # tempo mínimo em segundos para considerar presença na área

LOG_LEVEL = 2 # 0 = silencioso, 1 = normal, 2 = somente alertas
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
            # Calcula a caixa da área de interesse
            area_box = (
                AREA_X_MIN * frame.shape[1],
                AREA_Y_MIN * frame.shape[0],
                AREA_X_MAX * frame.shape[1],
                AREA_Y_MAX * frame.shape[0]
            )
            # Calcula interseção entre a caixa detectada e a área de interesse
            xA = max(x1, area_box[0])
            yA = max(y1, area_box[1])
            xB = min(x2, area_box[2])
            yB = min(y2, area_box[3])
            inter_area = max(0, xB - xA) * max(0, yB - yA)
            box_area = (x2 - x1) * (y2 - y1)
            inside_area = inter_area > 0.1 * box_area  # pelo menos 10% da caixa dentro da área

            if obj_id not in active_objects:
                active_objects[obj_id] = {
                    "label": model.names[int(box.cls[0])],
                    "last_seen": now,
                    "entry_time": now,
                    "logged_exit": False,
                    "alerted_level": 0,
                    "area_entry_time": None,
                    "area_last_inside": None,
                    "total_area_time": timedelta(0),  # Tempo total acumulado na área
                    "last_area_exit": None,  # Último momento que saiu da área
                    "is_in_area": False  # Flag para controlar estado atual na área
                }
                label = active_objects[obj_id]["label"]
                log(1, f"ID: {obj_id} - {label} ENTROU às {now.strftime('%H:%M:%S')}")
            else:
                active_objects[obj_id]["last_seen"] = now
                active_objects[obj_id]["logged_exit"] = False

            # Lógica atualizada para rastreamento na área
            if inside_area:
                if not active_objects[obj_id]["is_in_area"]:
                    # Objeto acabou de entrar na área
                    active_objects[obj_id]["is_in_area"] = True
                    active_objects[obj_id]["area_entry_time"] = now
                active_objects[obj_id]["area_last_inside"] = now
                active_objects[obj_id]["last_area_exit"] = None
            else:
                if active_objects[obj_id]["is_in_area"]:
                    # Objeto acabou de sair da área
                    active_objects[obj_id]["is_in_area"] = False
                    active_objects[obj_id]["last_area_exit"] = now
                    # Acumula o tempo que ficou na área
                    if active_objects[obj_id]["area_entry_time"]:
                        time_in_area = now - active_objects[obj_id]["area_entry_time"]
                        active_objects[obj_id]["total_area_time"] += time_in_area
                        active_objects[obj_id]["area_entry_time"] = None

    # Limpa objetos que saíram da câmera
    for oid, data in list(active_objects.items()):
        if now - data["last_seen"] > timedelta(seconds=TIMEOUT_SECONDS):
            if not data["logged_exit"]:
                log(1, f"ID: {oid} - {data['label']} SAIU às {now.strftime('%H:%M:%S')}")
                if data["total_area_time"].total_seconds() > 0:
                    log(1, f"ID: {oid} - {data['label']} permaneceu {data['total_area_time'].total_seconds():.1f}s na área")
                data["logged_exit"] = True
            del active_objects[oid]
            continue

        # Verifica se objeto está fora da área por muito tempo
        if not data["is_in_area"] and data["last_area_exit"]:
            time_outside = now - data["last_area_exit"]
            if time_outside.total_seconds() > AREA_TIMEOUT_SECONDS:
                # Se ficou fora da área por muito tempo, reseta o tempo total
                if data["total_area_time"].total_seconds() > 0:
                    log(1, f"ID: {oid} - {data['label']} saiu da área após {data['total_area_time'].total_seconds():.1f}s")
                data["total_area_time"] = timedelta(0)
                data["last_area_exit"] = None

        # Alerta para objetos que permanecem na área
        if data["is_in_area"] and data["area_entry_time"]:
            time_in_current_session = now - data["area_entry_time"]
            total_time = time_in_current_session + data["total_area_time"]
            if total_time.total_seconds() > AREA_PRESENCE_THRESHOLD and data["alerted_level"] < 2:
                log(2, f"ID {oid} - {data['label']} está há {total_time.total_seconds():.1f}s na área! ({now.strftime('%H:%M:%S')})")
                data["alerted_level"] = 2

    # Exibe resultado com anotações
    annotated = results[0].plot()
    # Desenhar área de interesse
    h, w, _ = frame.shape
    x1, y1 = int(AREA_X_MIN * w), int(AREA_Y_MIN * h)
    x2, y2 = int(AREA_X_MAX * w), int(AREA_Y_MAX * h)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow("Detecção - YOLOv8", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()