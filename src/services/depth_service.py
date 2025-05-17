import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

class DepthService:
    def __init__(self):
        # Carrega o modelo MiDaS
        self.model_type = "DPT_Large"     # MiDaS v3 - Large (mais preciso)
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        
        # Move o modelo para GPU se disponível
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        
        # Configuração das transformações
        self.midas_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Referência de profundidade
        self.reference_depth = None
        self.reference_frame = None
        self.depth_scale = None

    def calibrate_depth(self, frame):
        """
        Calibra a profundidade usando um frame de referência
        Deve ser chamado apenas uma vez no início
        """
        # Converte o frame para RGB (MiDaS espera RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Prepara a imagem para o modelo
        input_batch = self.midas_transforms(Image.fromarray(frame_rgb)).unsqueeze(0)
        input_batch = input_batch.to(self.device)
        
        # Faz a predição
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
        
        # Salva o mapa de profundidade de referência
        self.reference_depth = prediction.cpu().numpy()
        self.reference_frame = frame
        
        # Calcula a escala de profundidade baseada na altura do frame
        # Assumindo que objetos na parte inferior estão mais próximos
        h, w = frame.shape[:2]
        bottom_depth = np.mean(self.reference_depth[h-100:h, :])
        top_depth = np.mean(self.reference_depth[:100, :])
        self.depth_scale = (bottom_depth - top_depth) / h
        
        # Limpa recursos do modelo após calibração
        del self.midas
        torch.cuda.empty_cache()

    def estimate_depth_from_position(self, y_position):
        """
        Estima a profundidade baseada na posição Y do objeto
        y_position: posição Y do objeto no frame
        Retorna: valor de profundidade normalizado (0-1)
        """
        if self.depth_scale is None:
            return 0.5  # valor padrão se não houver calibração
        
        # Calcula a profundidade baseada na posição Y
        # Quanto maior o Y, mais próximo o objeto está
        h = self.reference_frame.shape[0]
        normalized_y = y_position / h
        depth = 1.0 - normalized_y  # inverte para que 0 seja mais próximo
        
        return depth

    def get_depth_for_box(self, box):
        """
        Retorna a profundidade estimada para uma região (bounding box)
        box: (x1, y1, x2, y2)
        Retorna: profundidade estimada (0-1)
        """
        if self.depth_scale is None:
            return 0.5  # valor padrão se não houver calibração
            
        x1, y1, x2, y2 = map(int, box)
        # Usa a posição Y do centro do objeto para estimar a profundidade
        center_y = (y1 + y2) / 2
        return self.estimate_depth_from_position(center_y)

    def cleanup(self):
        """Limpa recursos"""
        self.reference_depth = None
        self.reference_frame = None
        self.depth_scale = None
        torch.cuda.empty_cache() 