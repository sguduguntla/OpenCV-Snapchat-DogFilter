import numpy as np
import cv2
from video import create_capture

# Once the camera has detected two faces, it will swap the bounding box of the faces. Rudimentary face swap app!
# There are many improvements that should be made for it to be a good
# looking face swap


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(
        30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def get_cam_frame(cam):
    ret, img = cam.read()
    # smaller frame size - things run a lot smoother than a full screen img
    img = cv2.resize(img, (800, 450))
    return img


def main():
    # Camera 0 is usually the built in webcam camera... also most people only
    # have 1 webcam on their laptop
    cam = create_capture(0)
    # Loads images with grayscale
    dog_img = cv2.imread("dogfilter.png", cv2.CV_LOAD_IMAGE_COLOR)
    dalmatian_img = cv2.imread("dalmatianfilter.png", cv2.CV_LOAD_IMAGE_COLOR)

    # lbp_classifier = "../data/lbpcascade_frontalface.xml"
    haar_classifier = "../data/haarcascade_frontalface_default.xml"

    # use the haar classifier for now, it seems to work a little bit better
    cascade = cv2.CascadeClassifier(haar_classifier)

    while True:
        img = get_cam_frame(cam)

       # classifier wants things in black and white
        bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bw = cv2.equalizeHist(bw)

        rects = detect(bw, cascade)
        final = img.copy()

        # Mostly useful for debugging
        # for x1, y1, x2, y2 in rects:
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        # Have we detected at least 2 faces?
        if len(rects) >= 1:
            # A better solution would be to get the 2 largest rectangles from the list, because
            # sometimes we get false positives
            allFaces = rects

            #rect = (ax1, ay1, ax2, ay2)
            for index, (ax1, ay1, ax2, ay2) in enumerate(allFaces):
                deltaY = abs(ay2) - abs(ay1)
                stretchFactor = 0.2

                stretchAmount = int(stretchFactor * deltaY)

                ay2 = ay2 + stretchAmount
                ay1 = ay1 - stretchAmount

                height, width, _ = img.shape

                if ax1 > stretchAmount and ax2 < width - stretchAmount and ay1 > stretchAmount and ay2 < height - stretchAmount:
                    face = img[ay1:ay2, ax1:ax2]

                    dog_scale = []

                    if index % 2 == 0:
                        dog_scale = cv2.resize(dog_img, (ax2 - ax1, ay2 - ay1))
                    else:
                        dog_scale = cv2.resize(
                            dalmatian_img, (ax2 - ax1, ay2 - ay1))

                    my_scaled = np.where(dog_scale == 0, face, dog_scale)
                    # faceB = cv2.resize(
                    # img[by1:by2, bx1:bx2].copy(), (ax2 - ax1, ay2 - ay1))
                    final[ay1:ay2, ax1:ax2] = my_scaled
                    #final[by1:by2, bx1:bx2] = faceA

        cv2.imshow('face swap', final)

        # Esc key quits
        if 0xFF & cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
