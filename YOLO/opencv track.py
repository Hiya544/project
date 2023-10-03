import cv2 as cv
import numpy as np

# YOLO 로드
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[int(i[0]) - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 비디오 가져오기
video = cv.VideoCapture('test.mp4')

selected_object = None
object_selected = False
start_x, start_y = -1, -1  # 시작 좌표 초기화

# 객체 추적기 초기화 (KCF tracker 사용)
tracker = None

# 마우스 이벤트 콜백 함수
def mouse_event(event, x, y, flags, param):
    global selected_object, object_selected, start_x, start_y, tracker
    if event == cv.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y  # 객체 선택을 시작하는 지점 저장
        selected_object = None
        object_selected = False
        tracker = None
    elif event == cv.EVENT_LBUTTONUP:
        # 객체 선택 영역 계산
        end_x, end_y = x, y
        x, y, w, h = start_x, start_y, end_x - start_x, end_y - start_y
        selected_object = (x, y, w, h)
        object_selected = True
        tracker = cv.TrackerKCF_create()
        tracker.init(img, (x, y, w, h))  # 선택한 객체를 추적기로 초기화

cv.namedWindow('object_detection', cv.WINDOW_NORMAL)
cv.setMouseCallback('object_detection', mouse_event)

while video.isOpened():
    ret, img = video.read()
    if not ret:
        break

    img = cv.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    if object_selected and selected_object is not None:
        success, bbox = tracker.update(img)
        if success:
            x, y, w, h = map(int, bbox)
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            object_selected = False  # 추적이 끊겼을 때 객체 선택 해제

    else:
        # 객체가 선택되지 않은 경우, YOLO로 객체를 검출하고 선택합니다.
        blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 객체 선택 영역 선택
        if len(boxes) > 0:
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                if x < start_x < x + w and y < start_y < y + h:
                    selected_object = (x, y, w, h)
                    object_selected = True
                    tracker = cv.TrackerKCF_create()
                    tracker.init(img, (x, y, w, h))  # 선택한 객체를 추적기로 초기화

    cv.imshow('object_detection', img)
    key = cv.waitKey(1)
    if key == 27:  # ESC
        break

cv.destroyAllWindows()
