import cv2
import pytesseract

from app.east_preprocess import east_process
from app.get_text_area import get_masked
from app.mser_preprocess import mser_process


def ocr_text_generator(img):
    ocr_result = pytesseract.image_to_string(
        img, lang='eng', config='--psm 10 --oem 3')
    print(ocr_result)


if __name__ == "__main__":
    img = cv2.imread("./img_7.png")

    east_text_area = east_process(img)
    mser_text_area = mser_process(img)

    east_masked = get_masked(img, east_text_area)
    mser_masked = get_masked(img, mser_text_area)

    masked = cv2.bitwise_and(east_masked, mser_masked)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    ocr_text_generator(masked)
