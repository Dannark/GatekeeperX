from datetime import datetime, timedelta
import numpy as np
from src.config.settings import (
    SPEED_HISTORY_SIZE, 
    TRAJECTORY_HISTORY_SIZE, 
    DIRECTION_SMOOTHING_FACTOR,
    MIN_SPEED_THRESHOLD,
    MAX_SPEED_THRESHOLD,
    FRAME_HEIGHT,
    SPEED_CALIBRATION
)

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
        self.last_depth = 0.0  # profundidade do objeto

        # Trajetória tracking
        self.position_history = [position]  # Lista de tuplas (x, y)
        self.direction = (0, 0)  # Vetor de direção normalizado
        self.smoothed_direction = (0, 0)  # Vetor de direção suavizado
        self.movement_angle = 0  # Ângulo do movimento em graus
        self.direction_history = []  # Histórico de direções para suavização

    def update_speed(self, current_position, time_diff, frame_width, depth):
        """
        Atualiza a velocidade do objeto considerando a profundidade
        depth: valor de profundidade do objeto (0-1, onde 0 é mais próximo)
        """
        from src.utils.helpers import calculate_distance, calculate_speed
        
        distance = calculate_distance(self.last_position, current_position)
        speed_pixels = calculate_speed(distance, time_diff)
        
        # Ajusta a velocidade baseada na profundidade
        # Quanto menor a profundidade (mais próximo), maior a redução
        depth_factor = 0.3 + (depth * 0.7)  # Reduz mais a velocidade de objetos próximos
        adjusted_speed = speed_pixels * depth_factor
        
        # Converte para km/h
        # Assumindo que 100 pixels/s = SPEED_CALIBRATION km/h para objetos próximos
        speed_kmh = (adjusted_speed / 100) * SPEED_CALIBRATION
        
        # Aplica limites de velocidade
        if speed_kmh < MIN_SPEED_THRESHOLD:
            speed_kmh = 0
        elif speed_kmh > MAX_SPEED_THRESHOLD:
            speed_kmh = MAX_SPEED_THRESHOLD
        
        # Atualiza histórico
        self.speed_history.append(speed_kmh)
        if len(self.speed_history) > SPEED_HISTORY_SIZE:
            self.speed_history.pop(0)
        
        # Calcula média
        self.last_speed = sum(self.speed_history) / len(self.speed_history)
        self.last_position = current_position
        self.last_depth = depth
        self.last_speed_update = datetime.now()

    def update_trajectory(self, current_position):
        """Atualiza a trajetória e calcula a direção do movimento"""
        # Adiciona nova posição ao histórico
        self.position_history.append(current_position)
        if len(self.position_history) > TRAJECTORY_HISTORY_SIZE:
            self.position_history.pop(0)

        # Calcula direção apenas se tiver histórico suficiente
        if len(self.position_history) >= 2:
            # Pega as últimas duas posições para calcular a direção
            prev_pos = self.position_history[-2]
            curr_pos = self.position_history[-1]
            
            # Calcula o vetor de direção
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            
            # Normaliza o vetor
            magnitude = np.sqrt(dx*dx + dy*dy)
            if magnitude > 0:
                new_direction = (dx/magnitude, dy/magnitude)
                self.direction = new_direction
                
                # Atualiza histórico de direções
                self.direction_history.append(new_direction)
                if len(self.direction_history) > TRAJECTORY_HISTORY_SIZE:
                    self.direction_history.pop(0)
                
                # Calcula direção suavizada usando média móvel
                if len(self.direction_history) > 1:
                    avg_dx = sum(d[0] for d in self.direction_history) / len(self.direction_history)
                    avg_dy = sum(d[1] for d in self.direction_history) / len(self.direction_history)
                    
                    # Normaliza a direção suavizada
                    avg_magnitude = np.sqrt(avg_dx*avg_dx + avg_dy*avg_dy)
                    if avg_magnitude > 0:
                        self.smoothed_direction = (avg_dx/avg_magnitude, avg_dy/avg_magnitude)
                    else:
                        self.smoothed_direction = new_direction
                else:
                    self.smoothed_direction = new_direction
                
                # Calcula o ângulo em graus
                self.movement_angle = np.degrees(np.arctan2(dy, dx))

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