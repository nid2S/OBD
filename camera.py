import cv2

# camera setting


def camera():
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
    cv2.imwrite('./images/capture_image.jpg', frame)
