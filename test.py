import cv2
from ultralytics import YOLO
import numpy as np


# Load YOLOv8 model
model = YOLO('yolo12s.pt')
names = model.names


# Open video
cap = cv2.VideoCapture("wrongside.mp4")



# Debug mouse position
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)


frame_count = 0
area1=[(297,316),(288,355),(526,339),(518,299)]
area2=[(284,364),(269,404),(535,389),(523,346)]


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020,600))
    results = model.track(frame, persist=True,classes=[3])

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            label = names[class_id]
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1 + 3, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                 

      
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,0,255),2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(0) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
