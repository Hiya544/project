import cv2 as cv
import numpy as np

net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


video = cv.VideoCapture(1)

count = 0
while video.isOpened():
    ret, img = video.read()
    if not ret:
        break
    if int(video.get(1)) % 7 == 0:
        img = cv.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

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

        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]]) if len(class_ids) > 0 else 'No Object Detected'
                color = colors[class_ids[i]] if len(class_ids) > 0 else (0, 0, 0)
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv.putText(img, label, (x, y + 30), font, 3, color, 3)

        cv.imshow('object_detection', img)
        key = cv.waitKey(1)
        if key == 32:  
            key = cv.waitKey(0)
        if key == 27:  
            break
cv.destroyAllWindows()