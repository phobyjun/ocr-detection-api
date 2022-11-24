import math

import cv2


def decodeBoundingBoxes(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            if (score < scoreThresh):
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            offset = (
                [offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    return [detections, confidences]


def east_process(img):
    layerNames = ["feature_fusion/Conv_7/Sigmoid",
                  "feature_fusion/concat_3"]

    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    width = 640
    height = 640

    (origH, origW) = img.shape[:2]
    (newW, newH) = (width, height)
    rW = origW / float(newW)
    rH = origH / float(newH)

    img = cv2.resize(img, (newW, newH))
    (H, W) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    [boxes, confidences] = decodeBoundingBoxes(scores, geometry, 0.2)
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, 0.2, 0.7)

    text_area = []
    for i in indices:
        vertices = cv2.boxPoints(boxes[i])
        print(vertices)
        for j in range(4):
            vertices[j][0] *= rW
            vertices[j][1] *= rH

        for j in range(0, 2):
            p1 = (int(vertices[j][0]), int(vertices[j][1]))
            p2 = (int(vertices[j + 2][0]), int(vertices[j + 2][1]))
            text_area.append((p1, p2))

    return text_area


if __name__ == "__main__":
    img = cv2.imread("img_4.png")
    text_area = east_process(img)

    for area in text_area:
        p1, p2 = area
        cv2.rectangle(img, p1, p2, (255, 0, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
