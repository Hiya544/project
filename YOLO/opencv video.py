import cv2 as cv
import numpy as np
from tkinter import Tk, filedialog

def draw_path(path, frame, object_index):
    if len(path[object_index]) > 1:
        for i in range(1, len(path[object_index])):
            cv.line(frame, path[object_index][i - 1], path[object_index][i], (0, 0, 255), 2)

def create_new_object(x, y):
    global object_count, object_paths, selected_trackers, selected_objects

    selected_objects = (x - 40, y - 90, 80, 200)
    selected_trackers = cv.TrackerKCF_create()
    selected_trackers.init(frame, selected_objects)
    object_count += 1
    object_paths[object_count] = [(x, y)] 

def on_mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        create_new_object(x, y)

net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
root.destroy()
video = cv.VideoCapture(file_path)

selected_trackers = None
selected_objects = None
object_count = 0
object_paths = {}

cv.namedWindow('Video')  # 마우스 콜백을 설정하기 전에 창을 만듭니다.
cv.setMouseCallback('Video', on_mouse_click)  # 'Video' 창에 대한 마우스 콜백을 설정합니다.
MAX_PATH_LENGTH = 100

while True:
    ret, frame = video.read()
    if not ret:
        break

    if object_count in object_paths and selected_trackers is not None:
        success, bbox = selected_trackers.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if len(object_paths[object_count]) > MAX_PATH_LENGTH:  # 경로 최대 길이 이상인 경우
                object_paths[object_count] = object_paths[object_count][-MAX_PATH_LENGTH:]  # 뒷쪽부터 삭제
            object_paths[object_count].append((x + w // 2, y + h // 2))
            draw_path(object_paths, frame, object_count)
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
                class_id = scores.argmax()
                confidence = float(scores[class_id])
                if confidence > 0.5:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    boxes.append([x, y, width, height])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv.FONT_HERSHEY_PLAIN
        for j in range(len(boxes)):
            if j in indexes:
                x, y, w, h = boxes[j]
                label = str(classes[class_ids[j]])
                color = colors[class_ids[j]] if class_ids[j] < len(colors) else (0, 0, 0)
                confidence = confidences[j]
                text = f'{label} {confidence:.2f}'
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv.putText(frame, text, (x, y + 30), font, 3, color, 3)

    cv.imshow('Video', frame) 

    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyAllWindows()
