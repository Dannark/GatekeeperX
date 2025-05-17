from ultralytics import YOLO
import cv2
from datetime import datetime, timedelta
import math

# Carrega modelo YOLOv8 (baixado automaticamente)
model = YOLO("yolov8n.pt")  # Você pode trocar por yolov8s.pt para mais precisão

rtsp_url = "rtsp://Dannark:23021994@192.168.0.104:554/stream1"
cap = cv2.VideoCapture(rtsp_url)

# Obtém FPS da câmera para cálculos de velocidade
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1/fps  # Tempo entre frames em segundos

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

def calculate_distance(point1, point2):
    """Calcula a distância euclidiana entre dois pontos"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def calculate_speed(distance, time):
    """Calcula a velocidade em pixels por segundo"""
    return distance / time if time > 0 else 0

def pixels_to_meters(pixels, frame_width, real_width_meters=5):
    """Converte pixels para metros (assumindo uma largura conhecida da cena)"""
    return (pixels * real_width_meters) / frame_width

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
            # Calcula o centro do objeto
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_position = (center_x, center_y)

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
                    "total_area_time": timedelta(0),
                    "last_area_exit": None,
                    "is_in_area": False,
                    "last_position": current_position,
                    "last_speed": 0,
                    "speed_history": [],  # Histórico de velocidades para média móvel
                    "last_speed_update": now
                }
                label = active_objects[obj_id]["label"]
                log(1, f"ID: {obj_id} - {label} ENTROU às {now.strftime('%H:%M:%S')}")
            else:
                # Calcula velocidade
                time_diff = (now - active_objects[obj_id]["last_speed_update"]).total_seconds()
                if time_diff >= frame_time:  # Atualiza velocidade a cada frame
                    distance = calculate_distance(active_objects[obj_id]["last_position"], current_position)
                    speed_pixels = calculate_speed(distance, time_diff)
                    
                    # Converte velocidade para metros por segundo
                    speed_mps = pixels_to_meters(speed_pixels, frame.shape[1])
                    speed_kmh = speed_mps * 3.6  # Converte para km/h
                    
                    # Atualiza histórico de velocidades (média móvel de 5 amostras)
                    active_objects[obj_id]["speed_history"].append(speed_kmh)
                    if len(active_objects[obj_id]["speed_history"]) > 5:
                        active_objects[obj_id]["speed_history"].pop(0)
                    
                    # Calcula velocidade média
                    avg_speed = sum(active_objects[obj_id]["speed_history"]) / len(active_objects[obj_id]["speed_history"])
                    active_objects[obj_id]["last_speed"] = avg_speed
                    active_objects[obj_id]["last_speed_update"] = now
                    active_objects[obj_id]["last_position"] = current_position

                active_objects[obj_id]["last_seen"] = now
                active_objects[obj_id]["logged_exit"] = False

            # Lógica atualizada para rastreamento na área
            if inside_area:
                if not active_objects[obj_id]["is_in_area"]:
                    active_objects[obj_id]["is_in_area"] = True
                    active_objects[obj_id]["area_entry_time"] = now
                active_objects[obj_id]["area_last_inside"] = now
                active_objects[obj_id]["last_area_exit"] = None
            else:
                if active_objects[obj_id]["is_in_area"]:
                    active_objects[obj_id]["is_in_area"] = False
                    active_objects[obj_id]["last_area_exit"] = now
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
                if data["last_speed"] > 0:
                    log(1, f"ID: {oid} - {data['label']} velocidade média: {data['last_speed']:.1f} km/h")
                data["logged_exit"] = True
            del active_objects[oid]
            continue

        # Verifica se objeto está fora da área por muito tempo
        if not data["is_in_area"] and data["last_area_exit"]:
            time_outside = now - data["last_area_exit"]
            if time_outside.total_seconds() > AREA_TIMEOUT_SECONDS:
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
    
    # Desenha área de interesse
    h, w, _ = frame.shape
    x1, y1 = int(AREA_X_MIN * w), int(AREA_Y_MIN * h)
    x2, y2 = int(AREA_X_MAX * w), int(AREA_Y_MAX * h)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Adiciona informações de velocidade na tela
    for oid, data in active_objects.items():
        if oid in current_ids:
            for box in results[0].boxes:
                if int(box.id[0]) == oid:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    speed_text = f"{data['last_speed']:.1f} km/h"
                    cv2.putText(annotated, speed_text, (x1, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    break

    cv2.imshow("Detecção - YOLOv8", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()