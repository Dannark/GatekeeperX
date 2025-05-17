import math
from datetime import datetime
from src.config.settings import LOG_LEVEL

def calculate_distance(point1, point2):
    """Calcula a distância euclidiana entre dois pontos"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def calculate_speed(distance, time):
    """Calcula a velocidade em pixels por segundo"""
    return distance / time if time > 0 else 0

def pixels_to_meters(pixels, frame_width, real_width_meters=5):
    """Converte pixels para metros (assumindo uma largura conhecida da cena)"""
    return (pixels * real_width_meters) / frame_width

def log(level, message):
    """Função de logging com níveis"""
    if level >= LOG_LEVEL:
        print(message)

def format_time(dt):
    """Formata datetime para string legível"""
    return dt.strftime('%H:%M:%S') 