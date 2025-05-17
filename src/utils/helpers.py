import math
from datetime import datetime
from src.config.settings import LOG_LEVEL

def calculate_distance(point1, point2):
    """Calcula a distância euclidiana entre dois pontos"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def calculate_speed(distance_pixels, time_seconds):
    """Calcula a velocidade em pixels por segundo"""
    if time_seconds <= 0:
        return 0
    return distance_pixels / time_seconds

def get_depth_correction_factor(y_position, frame_height):
    """
    Calcula o fator de correção baseado na posição vertical do objeto
    y_position: posição Y do objeto (0 = topo, frame_height = base)
    frame_height: altura total do frame
    """
    # Normaliza a posição Y para 0-1 (0 = topo, 1 = base)
    normalized_y = y_position / frame_height
    
    # Aplica uma função exponencial para corrigir a perspectiva
    # Objetos mais próximos (y maior) têm correção maior
    correction = math.exp(normalized_y * PERSPECTIVE_CORRECTION_FACTOR)
    
    return correction

def pixels_to_meters(pixels_per_second, frame_width, y_position, frame_height, calibration_factor=1.2):
    """
    Converte velocidade de pixels/s para metros/s usando correção de perspectiva
    pixels_per_second: velocidade em pixels por segundo
    frame_width: largura do frame
    y_position: posição Y do objeto (para correção de profundidade)
    frame_height: altura do frame
    calibration_factor: fator de calibração base
    """
    # Calcula o fator de correção baseado na profundidade
    depth_correction = get_depth_correction_factor(y_position, frame_height)
    
    # Aplica a correção de profundidade ao fator de calibração
    adjusted_calibration = calibration_factor * depth_correction
    
    # Converte para metros por segundo
    meters_per_pixel = (REAL_WIDTH_METERS / frame_width) * adjusted_calibration
    return pixels_per_second * meters_per_pixel

def log(level, message):
    """
    Função de log com níveis
    level: nível do log (0 = silencioso, 1 = normal, 2 = somente alertas)
    message: mensagem a ser exibida
    """
    if level >= LOG_LEVEL:  # Inverte a lógica para que níveis maiores mostrem menos logs
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

def format_time(dt):
    """Formata datetime para string legível"""
    return dt.strftime('%H:%M:%S') 