import cv2
from src.services.detection_service import DetectionService

def main():
    detection_service = DetectionService()
    
    try:
        while True:
            frame, results, now = detection_service.process_frame()
            if frame is None:
                break
                
            annotated = detection_service.draw_annotations(frame, results, now)
            cv2.imshow("GatekeeperX", annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        detection_service.cleanup()

if __name__ == "__main__":
    main() 