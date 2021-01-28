import numpy as np
import cv2


def distance_squared(src, dst):
    return ((dst[0] - src[0]) * (dst[0] - src[0])) + ((dst[1] - src[1]) * (dst[1] - src[1]))


def random_color():
    color = tuple(np.random.choice(range(256), size=3))
    return int(color[0]), int(color[1]), int(color[2])


class Models:
    def __init__(self):
        self.YOLOV3 = "yolov3"
        self.YOLOV3_TINY = "yolov3-tiny"
        self.YOLOV4 = "yolov4"
        self.YOLOV4_TINY = "yolov4-tiny"
        self.ENET_COCO = "enet-coco"

    def get_models(self):
        return [self.YOLOV3, self.YOLOV3_TINY, self.YOLOV4, self.YOLOV4_TINY, self.ENET_COCO]


class Detector:
    def __init__(self):
        self.nn = None
        self.layer_names = []
        self.video = None
        self.group_colors = []

    def initialize(self, model_name, video_path):
        self.nn = cv2.dnn.readNetFromDarknet('Models/' + model_name + '.cfg', 'Models/' + model_name + '.weights')
        self.nn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.nn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.layer_names = self.nn.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.nn.getUnconnectedOutLayers()]

        if video_path != "":
            self.video = cv2.VideoCapture(video_path)
        else:
            self.video = cv2.VideoCapture(0)

        self.group_colors = []
        for i in range(255):
            self.group_colors.append(random_color())

    def analyze_frame(self, current_frame_count):
        can_read, frame = self.video.read()

        if not can_read:
            return False, None
        if current_frame_count != 1 and current_frame_count % 4 != 0:
            return True, None

        height, width = (frame.shape[0], frame.shape[1])

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.nn.setInput(blob)
        layer_outputs = self.nn.forward(self.layer_names)

        boxes = []
        confidences = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.4 and classID == 0:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, box_width, box_height) = box.astype('int')
                    boxes.append([
                        int(centerX - (box_width / 2)),
                        int(centerY - (box_height / 2)),
                        int(box_width),
                        int(box_height)
                    ])
                    confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)

        if len(indices) <= 0:
            return True, frame

        indices_flattened = indices.flatten()
        people = []
        for i in indices_flattened:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            area = w * h
            # Structure: Area, Center X, Center Y, Width, Height, Top-Left X, Top-Left Y
            people.append([area, x + w // 2, y + h // 2, w, h, x, y])

        people = sorted(people)

        human_groups = []
        current_group_index = 0

        human_groups.append([people[0]])

        for i in range(1, len(people)):
            area_ratio = people[i][0] / people[i - 1][0]
            if area_ratio > 1.4:
                human_groups.append([people[i]])
                current_group_index += 1
            else:
                human_groups[current_group_index].append(people[i])

        for i in range(len(human_groups)):
            group_total_height = 0
            for j in range(len(human_groups[i])):
                area, cx, cy, w, h, x, y = human_groups[i][j]
                group_total_height += h
            group_avg_height = group_total_height / len(human_groups[i])
            group_avg_height *= group_avg_height
            for j in range(len(human_groups[i])):
                area_j, cx_j, cy_j, w_j, h_j, x_j, y_j = human_groups[i][j]
                for k in range(len(human_groups[i])):
                    if j == k:
                        continue
                    area_k, cx_k, cy_k, w_k, h_k, x_k, y_k = human_groups[i][k]
                    d = distance_squared((cx_j, cy_j), (cx_k, cy_k))
                    if d < group_avg_height and abs(h_j - h_k) < 35:
                        cv2.line(frame, (cx_j, cy_j), (cx_k, cy_k), self.group_colors[i], 4)
                        cv2.circle(frame, (cx_j, cy_j), 4, self.group_colors[i], 4)
                        cv2.circle(frame, (cx_k, cy_k), 4, self.group_colors[i], 4)

        return True, frame
