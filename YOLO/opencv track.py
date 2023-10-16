import cv2 as cv
import numpy as np

# YOLO 로드
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 웹캠 초기화 - 여러 대의 웹캠을 사용할 경우 각 웹캠에 대한 VideoCapture 객체를 생성합니다.
webcams = [cv.VideoCapture(0), cv.VideoCapture(1)]  # 여기서 0와 1은 웹캠의 인덱스입니다.

# 객체 추적기 초기화 (KCF tracker 사용)
trackers = [None] * len(webcams)

# 선택한 객체의 정보를 저장할 변수
selected_objects = [None] * len(webcams)
object_selected_flags = [False] * len(webcams)

# 선택한 객체의 추적기 초기화
selected_trackers = [None] * len(webcams)

while True:
    for i, webcam in enumerate(webcams):
        ret, frame = webcam.read()
        if not ret:
            break

        # 객체 추적 중인 경우
        if object_selected_flags[i] and selected_trackers[i] is not None:
            success, bbox = selected_trackers[i].update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 객체 추적 중이 아닌 경우
        else:
            blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        w = int(detection[2] * frame.shape[1])
                        h = int(detection[3] * frame.shape[0])
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv.FONT_HERSHEY_PLAIN
            for j in range(len(boxes)):
                if j in indexes:
                    x, y, w, h = boxes[j]
                    label = str(classes[class_ids[j]]) if len(class_ids) > 0 else 'No Object Detected'
                    color = colors[class_ids[j]] if len(class_ids) > 0 else (0, 0, 0)
                    confidence = confidences[j]
                    text = f'{label} {confidence:.2f}'
                    cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv.putText(frame, text, (x, y + 30), font, 3, color, 3)

        # 웹캠 이미지를 독립적인 창에 표시
        cv.imshow(f'object_detection_{i}', frame)

        key = cv.waitKey(1)

        # 객체 추적 시작/종료 토글 (키보드 's' 키를 눌러 토글)
        if key == ord('s'):
            if object_selected_flags[i]:
                object_selected_flags[i] = False

                # 객체 추적 종료, 추적기 해제
                selected_trackers[i] = None
            else:
                if len(boxes) > 0:
                    # 객체 추적 시작, 첫 번째 객체 선택
                    selected_objects[i] = boxes[0]
                    object_selected_flags[i] = True

                    # 객체 추적기 초기화
                    x, y, w, h = selected_objects[i]
                    bbox = (x, y, w, h)
                    selected_trackers[i] = cv.TrackerKCF_create()
                    selected_trackers[i].init(frame, bbox)

    if key == 27:  # ESC
        break

# 웹캠 리소스 해제
for webcam in webcams:
    webcam.release()

cv.destroyAllWindows()
