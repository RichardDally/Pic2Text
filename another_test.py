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


def get_hud_team_names(image):
    x = 350
    y = 0
    h = 30
    w = 600
    image = crop(image, x, y, h, w)
    image = get_grey(image)
    preview(image)
    return image


def get_left_team_score(image):
    x = 595
    y = 50
    h = 33
    w = 35
    image = crop(image, x, y, h, w)
    image = get_grey(image)
    preview(image)
    return image


def get_right_team_score(image):
    x = 706
    y = 53
    h = 33
    w = 35
    image = crop(image, x, y, h, w)
    image = get_grey(image)
    preview(image)
    return image


def get_hud_timer(image):
    """
    Best results: 11, 12, 13
    """
    x = 630
    y = 56
    h = 25
    w = 70
    image = crop(image, x, y, h, w)
    image = get_grey(image)
    preview(image)
    return image


def preprocess(image):
    # All banner
    x = 350
    y = 0
    h = 30
    w = 600

    # Most close to DARKZERO ESPORTS
    #x = 375
    #y = 0
    #h = 30
    #w = 160

    # FAZE CLAN
    #x = 820
    #y = 0
    #h = 30
    #w = 110

    image = crop(image, x, y, h, w)
    image = get_grey(image)
    #image = canny(image)
    preview(image)
    #save(image)
    return image


def brute_force_image_to_string(image):
    for arg in range(3, 14):
        image_to_string(image, arg)


def image_to_string(image, preferred_param):
    try:
        text = pytesseract.image_to_string(image, config=f"-l eng --oem 1 --psm {preferred_param}")
        if text:
            logger.info(f"Parsed text [{text}] with [{preferred_param}]")
        else:
            logger.warning(f"Empty string for [{preferred_param}]")
    except Exception as exception:
        logger.exception(exception)


if __name__ == "__main__":
    """
    TODO:
    Note that all screenshots have 1326x747 pixels.
    get_hud_timers and all other methods should be rewritten to adapt 1920x1080 ! 
    """
    img = cv2.imread("match_1.png")
    timer = get_hud_timer(img)
    left_score = get_left_team_score(img)
    right_score = get_right_team_score(img)

    image_to_string(timer, 11)
    image_to_string(left_score, 13)
    image_to_string(right_score, 13)

    #img = preprocess(img)
    #brute_force_image_to_string(img)
    #draw_rectangles(img)
