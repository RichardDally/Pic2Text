import cv2
import pytesseract
from loguru import logger


image = cv2.imread("test.png")

x = 350
y = 0
h = 40
w = 200

#crop_img = image[y:y+h, x:x+w]
#cv2.imwrite("cropped.png", crop_img)

#text = pytesseract.image_to_string(crop_img, lang='eng')

image_to_process = image

for arg in range(3, 14):
    try:
        text = pytesseract.image_to_string(image_to_process, config=f"-l eng --oem 1 --psm {arg}")
        logger.info(f"Parsed text [{text}] with [{arg}]")
    except Exception as exception:
        logger.exception(exception)


#print(text)
