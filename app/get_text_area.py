import cv2

from east_preprocess import east_process
from mser_preprocess import mser_process


def get_mask(img, text_area):
    mask = img.copy()
    for area in text_area:
        p1, p2 = area
        copy = mask[p1[1]:p2[1], p1[0]:p2[0]]
        copy.fill(0)
    return 255 - mask


if __name__ == "__main__":
    img = cv2.imread("img_2.jpeg")
    east_text_area = east_process(img)
    mser_text_area = mser_process(img)
    text_area = east_text_area + mser_text_area

    mask = get_mask(img, text_area)

    masked = cv2.bitwise_and(img, mask)

    cv2.imshow('masked', masked)
    cv2.waitKey(0)
