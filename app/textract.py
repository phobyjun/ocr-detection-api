import datetime
import re

import boto3


class NoDateInImageException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def process_text_detection(bucket, document):
    client = boto3.client('textract')

    response = client.detect_document_text(
        Document={'S3Object': {'Bucket': bucket, 'Name': document}})

    blocks = response['Blocks']

    line_result = []
    for block in blocks:
        if block['BlockType'] == 'LINE':
            line_result.append(block['Text'])

    return line_result


def text_to_date(text_list):
    date_text_list = []
    for text in text_list:
        if (m := re.compile(r'\d{4}.\d{2}.\d{2}').match(text)) is not None:
            date_text_list.append((m.group(), "."))

        if (m := re.compile(r'\d{4}\d{2}\d{2}').match(text)) is not None:
            date_text_list.append((m.group(), ""))

        if (m := re.compile(r'\d{4}\s\d{2}\s\d{2}').match(text)) is not None:
            date_text_list.append((m.group(), " "))

        if (m := re.compile(r'\d{4}-\d{2}-\d{2}').match(text)) is not None:
            date_text_list.append((m.group(), "-"))

        if (m := re.compile(r'\d{4}/\d{2}/\d{2}').match(text)) is not None:
            date_text_list.append((m.group(), "/"))

    date_list = []
    for date_text in date_text_list:
        y, m, d = list(map(int, (date_text[0].split(date_text[1]))))
        date_list.append(datetime.date(y, m, d))

    date_list.sort(key=lambda x: (-x.year, -x.month, -x.day))

    return date_list


def get_date_from_image(image):
    bucket = 'naeng-bu-test'
    line_result = process_text_detection(bucket, image)
    date_list = text_to_date(line_result)

    if len(date_list) == 0:
        raise NoDateInImageException("No Detected Date In Image")

    return date_list[0]


def main():
    image = 'img_10.jpg'
    expiration_date = get_date_from_image(image)

    print(expiration_date)


if __name__ == "__main__":
    main()
