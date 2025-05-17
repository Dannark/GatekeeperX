import cv2
import numpy as np
from datetime import datetime, timedelta
from ultralytics import YOLO
from src.models.tracked_object import TrackedObject
from src.utils.helpers import log
from src.services.depth_service import DepthService
from src.config.settings import (
    RTSP_URL, TIMEOUT_SECONDS, AREA_TIMEOUT_SECONDS,
    AREA_PRESENCE_THRESHOLD, AREA_X_MIN, AREA_X_MAX,
    AREA_Y_MIN, AREA_Y_MAX, ARROW_LENGTH, ARROW_COLOR,
    ARROW_THICKNESS, ENTRANCE_LINE_START_X, ENTRANCE_LINE_START_Y,
    ENTRANCE_LINE_END_X, ENTRANCE_LINE_END_Y, ENTRANCE_LINE_COLOR,
    ENTRANCE_LINE_THICKNESS, MIN_CONFIDENCE, 
    MIN_SPEED_THRESHOLD, MAX_SPEED_THRESHOLD,
    ARROW_TIP_LENGTH, ARROW_TIP_ANGLE, LOOK_AT_COLOR
)

class DetectionService:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.depth_service = DepthService()
        self.cap = cv2.VideoCapture(RTSP_URL)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_time = 1/self.fps
        self.active_objects = {}
        
        # Calibra a profundidade com o primeiro frame
        ret, frame = self.cap.read()
        if ret:
            self.depth_service.calibrate_depth(frame)
            # Volta o vídeo para o início
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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

    def draw_direction_arrow(self, frame, center, direction, length=ARROW_LENGTH, color=ARROW_COLOR):
        """Desenha uma seta indicando a direção do movimento"""
        end_x = int(center[0] + direction[0] * length)
        end_y = int(center[1] + direction[1] * length)
        cv2.arrowedLine(frame, 
                       (int(center[0]), int(center[1])),
                       (end_x, end_y),
                       color,
                       ARROW_THICKNESS,
                       tipLength=0.3)

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
                # Verifica se o box tem ID e se é válido
                if box.id is None or len(box.id) == 0:
                    continue
                    
                try:
                    obj_id = int(box.id[0])
                    confidence = float(box.conf[0])  # Obtém a confiança da detecção
                    
                    # Ignora detecções com baixa confiança
                    if confidence < 0.65:  # 65% de confiança
                        continue
                        
                except (ValueError, IndexError):
                    continue

                current_ids.add(obj_id)
                x1, y1, x2, y2 = box.xyxy[0]
                
                # Calcula a posição do objeto
                # Para pessoas, usa um ponto 10% acima dos pés
                # Para outros objetos, usa o centro do retângulo
                if self.model.names[int(box.cls[0])] == "person":
                    center_x = (x1 + x2) / 2
                    height = y2 - y1
                    center_y = y2 - (height * 0.1)  # 10% acima dos pés
                else:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                current_position = (center_x, center_y)

                # Obtém a profundidade do objeto
                depth = self.depth_service.get_depth_for_box((x1, y1, x2, y2))

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
                        obj.update_speed(current_position, time_diff, frame.shape[1], depth)
                    obj.update_trajectory(current_position)
                    is_interested, should_log = obj.update_interest_score(frame.shape[1], frame.shape[0])
                    if is_interested and should_log:
                        log(2, f"ID {obj_id} - {obj.label} mostrando interesse! Score: {obj.interest_score:.1f} | Distância: {obj.last_distance:.2f}")
                    obj.last_seen = now
                    obj.logged_exit = False

                inside_area = self.is_inside_area((x1, y1, x2, y2), area_box)
                self.active_objects[obj_id].update_area_status(inside_area, now)

        self.cleanup_objects(now)
        return frame, results, now

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

    def draw_annotations(self, frame, results, now):
        """Desenha anotações no frame"""
        if not results or len(results) == 0:
            return frame
            
        annotated = results[0].plot()
        
        # Desenha área de interesse
        h, w, _ = frame.shape
        x1, y1 = int(AREA_X_MIN * w), int(AREA_Y_MIN * h)
        x2, y2 = int(AREA_X_MAX * w), int(AREA_Y_MAX * h)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Desenha linha de entrada da casa
        entrance_start_x = int(ENTRANCE_LINE_START_X * w)
        entrance_start_y = int(ENTRANCE_LINE_START_Y * h)
        entrance_end_x = int(ENTRANCE_LINE_END_X * w)
        entrance_end_y = int(ENTRANCE_LINE_END_Y * h)
        cv2.line(
            annotated,
            (entrance_start_x, entrance_start_y),
            (entrance_end_x, entrance_end_y),
            ENTRANCE_LINE_COLOR,
            ENTRANCE_LINE_THICKNESS
        )

        # Adiciona informações de velocidade e direção
        for oid, obj in self.active_objects.items():
            for box in results[0].boxes:
                # Verifica se o box tem ID e se é válido
                if box.id is None or len(box.id) == 0:
                    continue
                    
                try:
                    box_id = int(box.id[0])
                    if box_id == oid:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Calcula o ponto de referência do objeto
                        if obj.label == "person":
                            center_x = (x1 + x2) / 2
                            height = y2 - y1
                            center_y = y2 - (height * 0.1)  # 10% acima dos pés
                        else:
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                        
                        # Verifica interesse na casa
                        is_looking = obj.check_look_at(w, h)
                        
                        # Desenha seta de direção apenas se a velocidade for maior que 1 km/h
                        if len(obj.position_history) >= 2 and obj.last_speed > 1.0:
                            arrow_color = LOOK_AT_COLOR if is_looking else ARROW_COLOR
                            self.draw_direction_arrow(annotated, (center_x, center_y), obj.smoothed_direction, color=arrow_color)
                        
                        # Desenha velocidade
                        speed_text = f"{obj.last_speed:.1f} km/h"
                        
                        # Adiciona a distância à linha (sempre mostra)
                        distance_text = f" [Dist: {obj.last_distance:.2f}]"
                        speed_text += distance_text
                        
                        if obj.is_looking_at:
                            speed_text += " (Olhando)"
                        if obj.is_interested:
                            score_text = f" [Score: {obj.interest_score:.1f}]"
                            if obj.interest_start_time:
                                duration = now - obj.interest_start_time
                                score_text += f" ({duration.total_seconds():.1f}s)"
                            speed_text += score_text
                        
                        # Calcula o tamanho do texto para criar o fundo
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        (text_width, text_height), _ = cv2.getTextSize(speed_text, font, font_scale, thickness)
                        
                        # Desenha o fundo do texto no topo do retângulo
                        padding = 5
                        cv2.rectangle(
                            annotated,
                            (x1, y1 - text_height - padding * 2 + 30),
                            (x1 + text_width + padding * 2, y1 + 30),
                            (0, 0, 0),  # Cor preta para o fundo
                            -1  # Preenche o retângulo
                        )
                        
                        # Desenha o texto da velocidade no topo
                        cv2.putText(
                            annotated,
                            speed_text,
                            (x1 + padding, y1 - padding + 30),
                            font,
                            font_scale,
                            (255, 255, 255),  # Cor branca para o texto
                            thickness
                        )
                        break
                except (ValueError, IndexError):
                    continue

        return annotated

    def cleanup(self):
        """Limpa recursos"""
        self.cap.release()
        self.depth_service.cleanup()
        cv2.destroyAllWindows() 