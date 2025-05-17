import cv2
from datetime import datetime, timedelta
from ultralytics import YOLO
from src.models.tracked_object import TrackedObject
from src.utils.helpers import log
from src.config.settings import (
    RTSP_URL, TIMEOUT_SECONDS, AREA_TIMEOUT_SECONDS,
    AREA_PRESENCE_THRESHOLD, AREA_X_MIN, AREA_X_MAX,
    AREA_Y_MIN, AREA_Y_MAX
)

class DetectionService:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(RTSP_URL)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_time = 1/self.fps
        self.active_objects = {}

    def calculate_area_box(self, frame_shape):
        """Calcula as coordenadas da área de interesse"""
        return (
            AREA_X_MIN * frame_shape[1],
            AREA_Y_MIN * frame_shape[0],
            AREA_X_MAX * frame_shape[1],
            AREA_Y_MAX * frame_shape[0]
        )

    def is_inside_area(self, box, area_box):
        """Verifica se o objeto está dentro da área de interesse"""
        x1, y1, x2, y2 = box
        xA = max(x1, area_box[0])
        yA = max(y1, area_box[1])
        xB = min(x2, area_box[2])
        yB = min(y2, area_box[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        box_area = (x2 - x1) * (y2 - y1)
        return inter_area > 0.1 * box_area

    def process_frame(self):
        """Processa um frame da câmera"""
        ret, frame = self.cap.read()
        if not ret:
            log(1, "Erro ao acessar o stream")
            return None

        results = self.model.track(source=frame, persist=True, conf=0.5, verbose=False)
        now = datetime.now()
        current_ids = set()
        area_box = self.calculate_area_box(frame.shape)

        for r in results:
            for box in r.boxes:
                obj_id = int(box.id[0]) if box.id is not None else None
                if obj_id is None:
                    continue

                current_ids.add(obj_id)
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                current_position = (center_x, center_y)

                if obj_id not in self.active_objects:
                    self.active_objects[obj_id] = TrackedObject(
                        obj_id,
                        self.model.names[int(box.cls[0])],
                        current_position
                    )
                    log(1, f"ID: {obj_id} - {self.active_objects[obj_id].label} ENTROU às {now.strftime('%H:%M:%S')}")
                else:
                    obj = self.active_objects[obj_id]
                    time_diff = (now - obj.last_speed_update).total_seconds()
                    if time_diff >= self.frame_time:
                        obj.update_speed(current_position, time_diff, frame.shape[1])
                    obj.last_seen = now
                    obj.logged_exit = False

                inside_area = self.is_inside_area((x1, y1, x2, y2), area_box)
                self.active_objects[obj_id].update_area_status(inside_area, now)

        self.cleanup_objects(now)
        return frame, results

    def cleanup_objects(self, now):
        """Limpa objetos que saíram da câmera ou da área"""
        for oid, obj in list(self.active_objects.items()):
            if now - obj.last_seen > timedelta(seconds=TIMEOUT_SECONDS):
                if not obj.logged_exit:
                    log(1, f"ID: {oid} - {obj.label} SAIU às {now.strftime('%H:%M:%S')}")
                    if obj.total_area_time.total_seconds() > 0:
                        log(1, f"ID: {oid} - {obj.label} permaneceu {obj.total_area_time.total_seconds():.1f}s na área")
                    if obj.last_speed > 0:
                        log(1, f"ID: {oid} - {obj.label} velocidade média: {obj.last_speed:.1f} km/h")
                    obj.logged_exit = True
                del self.active_objects[oid]
                continue

            if not obj.is_in_area and obj.last_area_exit:
                time_outside = now - obj.last_area_exit
                if time_outside.total_seconds() > AREA_TIMEOUT_SECONDS:
                    if obj.total_area_time.total_seconds() > 0:
                        log(1, f"ID: {oid} - {obj.label} saiu da área após {obj.total_area_time.total_seconds():.1f}s")
                    obj.total_area_time = timedelta(0)
                    obj.last_area_exit = None

            if obj.is_in_area and obj.area_entry_time:
                time_in_current_session = now - obj.area_entry_time
                total_time = time_in_current_session + obj.total_area_time
                if total_time.total_seconds() > AREA_PRESENCE_THRESHOLD and obj.alerted_level < 2:
                    log(2, f"ID {oid} - {obj.label} está há {total_time.total_seconds():.1f}s na área! ({now.strftime('%H:%M:%S')})")
                    obj.alerted_level = 2

    def draw_annotations(self, frame, results):
        """Desenha anotações no frame"""
        annotated = results[0].plot()
        
        # Desenha área de interesse
        h, w, _ = frame.shape
        x1, y1 = int(AREA_X_MIN * w), int(AREA_Y_MIN * h)
        x2, y2 = int(AREA_X_MAX * w), int(AREA_Y_MAX * h)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Adiciona informações de velocidade
        for oid, obj in self.active_objects.items():
            for box in results[0].boxes:
                if int(box.id[0]) == oid:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    speed_text = f"{obj.last_speed:.1f} km/h"
                    cv2.putText(annotated, speed_text, (x1, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    break

        return annotated

    def cleanup(self):
        """Limpa recursos"""
        self.cap.release()
        cv2.destroyAllWindows() 