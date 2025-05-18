# Configurações da câmera
RTSP_URL = "rtsp://Dannark:23021994@192.168.0.104:554/stream1"

# Configurações de detecção
MIN_CONFIDENCE = 0.65  # nível mínimo de confiança para considerar uma detecção válida

# Configurações de tempo
TIMEOUT_SECONDS = 3  # tolerância para considerar que saiu da cena
AREA_TIMEOUT_SECONDS = 3  # tolerância para considerar que saiu da área
AREA_PRESENCE_THRESHOLD = 10  # tempo mínimo em segundos para considerar presença na área

# Configurações de área de interesse (percentual da largura e altura)
AREA_X_MIN = 0.1
AREA_X_MAX = 0.9
AREA_Y_MIN = 0.55
AREA_Y_MAX = 0.95

# Configurações da linha de entrada da casa (percentual da largura e altura)
ENTRANCE_LINE_START_X = 0.1  # Ponto A - X inicial
ENTRANCE_LINE_START_Y = 0.55  # Ponto A - Y inicial
ENTRANCE_LINE_END_X = 0.2    # Ponto B - X final
ENTRANCE_LINE_END_Y = 0.95    # Ponto B - Y final
ENTRANCE_LINE_COLOR = (0, 0, 255)  # Cor vermelha (BGR)
ENTRANCE_LINE_THICKNESS = 2

# Configurações de tracking
SPEED_HISTORY_SIZE = 5  # tamanho da média móvel para cálculo de velocidade
TRAJECTORY_HISTORY_SIZE = 10  # número de posições para manter no histórico de trajetória
DIRECTION_SMOOTHING_FACTOR = 0.3  # fator de suavização da direção (0-1)

# Configurações de velocidade
MIN_SPEED_THRESHOLD = 0.5  # velocidade mínima em km/h para considerar movimento
MAX_SPEED_THRESHOLD = 30.0  # velocidade máxima em km/h para filtrar ruído
SPEED_CALIBRATION = 5.0  # velocidade de referência em km/h para 100 pixels/s

# Configurações de calibração de velocidade
PERSPECTIVE_CORRECTION_FACTOR = 2.0  # fator de correção da perspectiva (maior = mais correção)
FRAME_HEIGHT = 720  # altura do frame em pixels (ajuste conforme sua câmera)

# Configurações de visualização
ARROW_LENGTH = 30  # comprimento da seta de direção em pixels
ARROW_COLOR = (0, 255, 0)  # cor da seta (BGR)
ARROW_THICKNESS = 2  # espessura da seta
ARROW_TIP_LENGTH = 0.3  # comprimento da ponta da seta (proporção do comprimento total)
ARROW_TIP_ANGLE = 30  # ângulo da ponta da seta em graus

# Configurações de detecção de olhar
LOOK_AT_ANGLE_THRESHOLD = 30  # ângulo máximo em graus para considerar que está olhando para a casa
LOOK_AT_DISTANCE_THRESHOLD = 0.7  # distância máxima normalizada para considerar olhar (0-1)
LOOK_AT_COLOR = (0, 165, 255)  # cor laranja para indicar olhar (BGR)

# Configurações de pontuação de interesse
INTEREST_SCORE_THRESHOLD = 40  # pontuação mínima para considerar interesse
INTEREST_SCORE_LOOK_AT = 2  # pontos por frame olhando para a casa
INTEREST_SCORE_STANDING = 1  # pontos por frame parado próximo à casa
INTEREST_SCORE_DECAY = 0.95  # fator de decaimento base da pontuação por frame
INTEREST_DISTANCE_THRESHOLD = 0.3  # distância máxima normalizada para considerar próximo à casa (0-1)
INTEREST_SPEED_THRESHOLD = 1.0  # velocidade máxima em km/h para considerar parado

# Configurações de log
LOG_LEVEL = 2  # 0 = silencioso, 1 = normal, 2 = somente alertas 