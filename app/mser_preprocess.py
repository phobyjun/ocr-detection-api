import cv2


def mser_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions, boundingBoxes = mser.detectRegions(gray)

    text_area = []
    for box in boundingBoxes:
        x, y, w, h = box
        p1 = (x, y)
        p2 = ((x + w), (y + h))
        text_area.append((p1, p2))

    return text_area


if __name__ == "__main__":
    img = cv2.imread("zbar.png")
    text_area = mser_process(img)
