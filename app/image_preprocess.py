import cv2
import imutils
import numpy as np
import pytesseract
from imutils.contours import sort_contours


def area_grouping(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (H, W) = gray.shape

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 20))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (221, 50))

    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")

    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    close_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    close_thresh = cv2.erode(close_thresh, None, iterations=2)

    return close_thresh


def area_detecting(img, close_thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (H, W) = gray.shape

    cnts = cv2.findContours(close_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="top-to-bottom")[0]

    roi_list = []
    roi_title_list = []

    margin = 20
    receipt_grouping = img.copy()

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w // float(h)

        color = (0, 255, 0)
        roi = img[y - margin:y + h + margin, x - margin:x + w + margin]
        roi_list.append(roi)
        roi_title_list.append("Roi_{}".format(len(roi_list)))

        cv2.rectangle(receipt_grouping, (x - margin, y - margin), (x + w + margin, y + h + margin), color, 2)
        cv2.putText(receipt_grouping, "".join(str(ar)), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return receipt_grouping, roi_list


if __name__ == "__main__":
    img = cv2.imread("img_8.png")
    close_thresh = area_grouping(img)
    cv2.imshow('img', close_thresh)
    cv2.waitKey(0)

    receipt_grouping, roi_list = area_detecting(img, close_thresh)
    cv2.imshow('img', receipt_grouping)
    cv2.waitKey(0)

    for roi in roi_list:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        threshold_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        roi_text = pytesseract.image_to_string(threshold_roi)
        print("text: {}".format(roi_text))
