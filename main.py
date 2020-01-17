import cv2
import numpy
import pytesseract
from loguru import logger
from matplotlib import pyplot as plt


def remove_noise(image):
    return cv2.medianBlur(image,5)


def opening(image):
    kernel = numpy.ones((5,5), numpy.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def canny(image):
    return cv2.Canny(image, 100, 200)


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def get_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_thresh(image):
    return cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]


def save(image):
    cv2.imwrite("another_output.png", image)


def crop(image, x, y, h, w):
    return image[y:y+h, x:x+w]


def preview(image):
    plt.figure(figsize=(40, 40))
    plt.imshow(image, cmap="gray")
    plt.title('image')
    plt.show()


def draw_rectangles(image):
    h, w, c = image.shape
    boxes = pytesseract.image_to_boxes(image)
    logger.info(boxes)
    for b in boxes.splitlines():
        b = b.split(' ')
        image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    preview(image)


def get_left_team_name(image):
    """
    1920x1080 confirmed.
    Best results: 6, 7, 8, 10, 11, 12, 13
    """
    x = 650
    y = 0
    h = 45
    w = 130
    # preview(image)
    image = crop(image, x, y, h, w)
    image = get_grey(image)
    #preview(image)
    return image


def get_right_team_name(image):
    """
    1920x1080 confirmed.
    TO improve !
    Best results: 10, 11, 12
    """
    x = 1143
    y = 5
    h = 40
    w = 195
    # preview(image)
    image = crop(image, x, y, h, w)
    image = get_grey(image)
    image = get_thresh(image)
    #preview(image)
    return image


def get_left_team_score(image):
    """
    1920x1080 confirmed.
    Best results: 6, 7, 9, 10
    """
    x = 853
    y = 75
    h = 42
    w = 50
    #preview(image)
    image = crop(image, x, y, h, w)
    image = get_grey(image)
    #preview(image)
    return image


def get_right_team_score(image):
    """
    1920x1080 confirmed.
    Best results: 6, 7, 9, 10
    """
    x = 1017
    y = 75
    h = 42
    w = 50
    image = crop(image, x, y, h, w)
    image = get_grey(image)
    #preview(image)
    return image


def get_hud_timer(image):
    """
    1920x1080 confirmed.
    Best results: 6, 8, 13
    TODO: improve
    """
    x = 915
    y = 80
    h = 33
    w = 90
    image = crop(image, x, y, h, w)
    image = get_grey(image)
    #preview(image)
    return image


def brute_force_image_to_string(image):
    for arg in range(3, 14):
        image_to_string(image, arg)


def image_to_string(image, preferred_param):
    text = None
    try:
        text = pytesseract.image_to_string(image, config=f"-l eng --oem 1 --psm {preferred_param}")
    except Exception as exception:
        logger.exception(exception)
    return text


def extract_screenshot(filename):
    logger.info(f"Processing [{filename}]")
    img = cv2.imread(filename)

    # Crop + preprocess
    timer_image = get_hud_timer(img)
    left_score_image = get_left_team_score(img)
    right_score_image = get_right_team_score(img)
    left_name_image = get_left_team_name(img)
    right_name_image = get_right_team_name(img)

    # OCR
    timer = image_to_string(timer_image, 13)
    left_score = image_to_string(left_score_image, 10)
    right_score = image_to_string(right_score_image, 10)
    left_team_name = image_to_string(left_name_image, 13)
    right_team_name = image_to_string(right_name_image, 12)

    # Display extracted info
    logger.info(f"Match => [{left_team_name}] VS [{right_team_name}]")
    logger.info(f"Score [{left_score}] for [{left_team_name}]")
    logger.info(f"Score [{right_score}] for [{right_team_name}]")
    logger.info(f"Remaining time [{timer}]")


def test():
    img = cv2.imread("standard.png")
    timer = get_hud_timer(img)
    left_score_image = get_left_team_score(img)
    right_score_image = get_right_team_score(img)
    left_name_image = get_left_team_name(img)
    right_name_image = get_right_team_name(img)

    # brute_force_image_to_string(right_name)
    image_to_string(timer, 13)
    left_score = image_to_string(left_score_image, 10)
    right_score = image_to_string(right_score_image, 10)
    left_team_name = image_to_string(left_name_image, 13)
    right_team_name = image_to_string(right_name_image, 12)

    # img = preprocess(img)
    # brute_force_image_to_string(img)
    # draw_rectangles(img)


if __name__ == "__main__":
    for screenshot in ["1_kill.png", "3_kills.png", "start_of_round.png", "end_of_round.png", "standard.png"]:
        extract_screenshot(screenshot)
