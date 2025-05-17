# Configurações da câmera
RTSP_URL = "rtsp://Dannark:23021994@192.168.0.104:554/stream1"

# Configurações de tempo
TIMEOUT_SECONDS = 3  # tolerância para considerar que saiu da cena
AREA_TIMEOUT_SECONDS = 3  # tolerância para considerar que saiu da área
AREA_PRESENCE_THRESHOLD = 10  # tempo mínimo em segundos para considerar presença na área

# Configurações de área de interesse (percentual da largura e altura)
AREA_X_MIN = 0.1
AREA_X_MAX = 0.95
AREA_Y_MIN = 0.3
AREA_Y_MAX = 0.95

# Configurações de velocidade
REAL_WIDTH_METERS = 5  # largura real da cena em metros
SPEED_HISTORY_SIZE = 5  # tamanho da média móvel para cálculo de velocidade
TRAJECTORY_HISTORY_SIZE = 10  # número de posições para manter no histórico de trajetória
DIRECTION_SMOOTHING_FACTOR = 0.3  # fator de suavização da direção (0-1)

# Configurações de visualização
ARROW_LENGTH = 30  # comprimento da seta de direção em pixels
ARROW_COLOR = (0, 255, 0)  # cor da seta (BGR)
ARROW_THICKNESS = 2  # espessura da seta

# Configurações de log
LOG_LEVEL = 2  # 0 = silencioso, 1 = normal, 2 = somente alertas 