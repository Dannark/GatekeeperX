import cv2
from src.services.detection_service import DetectionService

def main():
    detection_service = DetectionService()
    
    try:
        while True:
            result = detection_service.process_frame()
            if result is None:
                break
                
            frame, results = result
            annotated = detection_service.draw_annotations(frame, results)
            
            cv2.imshow("Detecção - YOLOv8", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        detection_service.cleanup()

if __name__ == "__main__":
    main() 