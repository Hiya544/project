import cv2 as cv
import numpy as np


net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

webcams = [cv.VideoCapture(0), cv.VideoCapture(1)]  


trackers = [None] * len(webcams)


selected_objects = [None] * len(webcams)
object_selected_flags = [False] * len(webcams)


selected_trackers = [None] * len(webcams)

while True:
    for i, webcam in enumerate(webcams):
        ret, frame = webcam.read()
        if not ret:
            break

        
        if object_selected_flags[i] and selected_trackers[i] is not None:
            success, bbox = selected_trackers[i].update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        
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


        cv.imshow(f'object_detection_{i}', frame)

        key = cv.waitKey(1)

        if key == ord('s'):
            if object_selected_flags[i]:
                object_selected_flags[i] = False

                
                selected_trackers[i] = None
            else:
                if len(boxes) > 0:
                   
                    selected_objects[i] = boxes[0]
                    object_selected_flags[i] = True

                    
                    x, y, w, h = selected_objects[i]
                    bbox = (x, y, w, h)
                    selected_trackers[i] = cv.TrackerKCF_create()
                    selected_trackers[i].init(frame, bbox)

    if key == 27: 
        break


for webcam in webcams:
    webcam.release()

cv.destroyAllWindows()
