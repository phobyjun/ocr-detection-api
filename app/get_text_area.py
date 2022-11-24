import cv2

from east_preprocess import east_process
from mser_preprocess import mser_process


def get_masked(img, text_area):
    mask = img.copy()
    for area in text_area:
        p1, p2 = area
        copy = mask[p1[1] - 10:p2[1] + 10, p1[0] - 10:p2[0] + 10]
        copy.fill(0)

    mask = 255 - mask

    masked = cv2.bitwise_and(img, mask)

    return masked


if __name__ == "__main__":
    img = cv2.imread("img.jpeg")
    east_text_area = east_process(img)
    mser_text_area = mser_process(img)

    east_mask = get_masked(img, east_text_area)
    mser_mask = get_masked(img, mser_text_area)

    mask = cv2.bitwise_and(east_mask, mser_mask)

    masked = cv2.bitwise_and(img, mask)

    cv2.imshow('masked', masked)
    cv2.waitKey(0)
