from datetime import datetime, timedelta
from src.config.settings import SPEED_HISTORY_SIZE

class TrackedObject:
    def __init__(self, obj_id, label, position):
        self.id = obj_id
        self.label = label
        self.last_seen = datetime.now()
        self.entry_time = datetime.now()
        self.logged_exit = False
        self.alerted_level = 0
        
        # Área tracking
        self.area_entry_time = None
        self.area_last_inside = None
        self.total_area_time = timedelta(0)
        self.last_area_exit = None
        self.is_in_area = False
        
        # Velocidade tracking
        self.last_position = position
        self.last_speed = 0
        self.speed_history = []
        self.last_speed_update = datetime.now()

    def update_speed(self, current_position, time_diff, frame_width):
        """Atualiza a velocidade do objeto"""
        from src.utils.helpers import calculate_distance, calculate_speed, pixels_to_meters
        
        distance = calculate_distance(self.last_position, current_position)
        speed_pixels = calculate_speed(distance, time_diff)
        
        # Converte para km/h
        speed_mps = pixels_to_meters(speed_pixels, frame_width)
        speed_kmh = speed_mps * 3.6
        
        # Atualiza histórico
        self.speed_history.append(speed_kmh)
        if len(self.speed_history) > SPEED_HISTORY_SIZE:
            self.speed_history.pop(0)
        
        # Calcula média
        self.last_speed = sum(self.speed_history) / len(self.speed_history)
        self.last_position = current_position
        self.last_speed_update = datetime.now()

    def update_area_status(self, is_inside, now):
        """Atualiza o status do objeto na área"""
        if is_inside:
            if not self.is_in_area:
                self.is_in_area = True
                self.area_entry_time = now
            self.area_last_inside = now
            self.last_area_exit = None
        else:
            if self.is_in_area:
                self.is_in_area = False
                self.last_area_exit = now
                if self.area_entry_time:
                    time_in_area = now - self.area_entry_time
                    self.total_area_time += time_in_area
                    self.area_entry_time = None 