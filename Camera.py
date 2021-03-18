import cv2
import numpy as np
from PIL import Image, ImageOps


def camera(order=0):
    videocapture = cv2.VideoCapture(0)

    if videocapture.isOpened() is False:
        print("Could not open the Camera.")
        return

    while True:
        exist, frame = videocapture.read()

        if exist == "false":
            print("Video was closed.")
            break

        cv2.imshow('camera', frame)

        # ASCII / 10 > Enter / 27 > ESC / 32 > Space
        # if push the space bar, frame is saved, camera is ended.
        if cv2.waitKey(1) & 0xFF == 32:
            cv2.destroyAllWindows()
            break

    # save a frame in images folder
    cv2.imwrite(f'./images/capture{order}.jpg', frame)


def pilImread(size, link='./images/capture0.jpg'):
    # return image(ndarray) of shape (1, y, x).

    image = Image.open(link)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = image.astype(np.float32)/255.0
    image = np.expand_dims(image, 0)

    return image


def cv2Imread(size, link='./images/capture0.jpg'):
    # return image(ndarray) of shape (1, y, x, 3).

    image = cv2.imread(link)
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, 0)

    return image

