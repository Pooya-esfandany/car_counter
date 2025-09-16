
from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import time
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def chack_line_collision(line1, line2):
    A, B = line1
    C, D = line2
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
cap = cv.VideoCapture("cars.mp4")  # For Video
model = YOLO("yolov8l.pt")
mask = cv.imread("mask.png")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
def eventcallback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print("x="+str(x) + ", y=" + str(y))
prev_frame_time = 0
new_frame_time = 0
added = []
line = (260,306),(732,323)
while True:
    new_frame_time = time.time()
    success, frame = cap.read()
    region  = cv.bitwise_and(frame,mask)
    results = model.track(region, persist=True, tracker="bytetrack.yaml",verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w,h = (x1-x2), (y1-y2)
            cv.rectangle(frame, (x2, y2), (x1, y1), (255, 0, 255), 3)
            index = int(box.cls[0])
            string = classNames[index]
            id =str(int(box.id[0].item()))
            cv.putText(frame,string,(x1,y1),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255))
            line1 = (x1,y1),(x1,y2)
            line2 = (x1,y2),(x2,y2)
            line3 = (x2,y1),(x2,y2)
            line4 = (x2,y2),(x2,y1)
            passed_line = False
            if chack_line_collision(line1, line) or chack_line_collision(line2, line) or chack_line_collision(line3, line) or chack_line_collision(line4, line):
                if id in added:
                    pass
                else:
                    added.append(id)
            if id in added:
                cv.rectangle(frame, (x2, y2), (x1, y1), (0, 255, 0), 3)
            else:
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv.line(frame, line[0], line[1], (255, 255, 255), 2, lineType=cv.LINE_AA)
    cv.imshow("Image", frame)
    cv.setMouseCallback("Image", eventcallback)

    cv.waitKey(30)
