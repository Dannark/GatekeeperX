import cv2

rtsp_url = "rtsp://Dannark:23021994@192.168.0.102:554/stream1"
cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar o stream")
        break
    cv2.imshow("Câmera Tapo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()