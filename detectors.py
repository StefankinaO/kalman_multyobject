'''
    File name         : detectors.py
    File Description  : Detect objects in video frame
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
import cv2
import tensorflow as tf
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import glob

# set to 1 for pipeline images
debug = 0


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
        path: the file path to the image

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    print(path)
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    image = image.convert('RGB')
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


class Detectors(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """

    def __init__(self):
        """Initialize variables used by Detectors class
        Args:
            None
        Return:
            None
        """
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def Detect(self, frame):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """

        # Convert BGR to GRAY

        '''if (debug == 1):
            cv2.imshow('gray', frame)

        # Perform Background Subtraction
        fgmask = self.fgbg.apply(frame)

        if (debug == 0):
            cv2.imshow('bgsub', fgmask)

        # Detect edges
        edges = cv2.Canny(fgmask, 50, 190, 3)

        if (debug == 1):
            cv2.imshow('Edges', edges)

        # Retain only edges within the threshold
        ret, thresh = cv2.threshold(edges, 127, 255, 0)

        # Find contours
        contours, hierarchy = cv2.findContours(thresh,
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)

        if (debug == 0):
            cv2.imshow('thresh', thresh)

        centers = []  # vector of object centroids in a frame
        # we only care about centroids with size of bug in this example
        # recommended to be tunned based on expected object size for
        # improved performance
        blob_radius_thresh = 8
        # Find centroid for each valid contours
        for cnt in contours:
            try:
                # Calculate and draw circle
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                centeroid = (int(x), int(y))
                radius = int(radius)
                if (radius > blob_radius_thresh):
                    cv2.circle(frame, centeroid, radius, (0, 255, 0), 2)
                    b = np.array([[x], [y]])
                    centers.append(np.round(b))
            except ZeroDivisionError:
                pass

        # show contours of tracking objects
        #cv2.imshow('Track Bugs', frame)'''
        centers = []
        new_det = []
        f = open('myfile.txt')
        stroka = f.read()
        stroka = stroka.replace("dtype=float32", "")
        f.close()
        for i in range(stroka.count("))") + 1):
            if "283_" + str(i) + ".png" in stroka:
                new_det.append(stroka[stroka.index("[[") + 1: stroka.index("]]") + 1])
                try:
                    stroka = stroka[len(stroka[stroka.index("283_" + str(i) + ".png") - 2: stroka.index(")),") + 4]):]
                except:
                    break
        itog = []
        promesh = ''
        res = []
        for i in range(len(new_det)):
            for j in range(len(new_det[i])):
                if new_det[i][j] != ',' and new_det[i][j] != '[' and new_det[i][j] != ']':
                    promesh += new_det[i][j]
                if (new_det[i][j] == ',' or new_det[i][j] == ']') and (len(promesh) != 0):
                    itog.append(float(promesh) * 612)
                    promesh = ''

            res.append(itog)
            itog = []

        centers = []
        k = 0
        X = []
        Y = []
        z = 0
        print(len(res))
        for i in range(len(res)):
            for j in range(len(res[0]) - 3):
                #[ymin, xmin, ymax, xmax]
                #(x1 + (x2 - x1) / 2, y2 + (y1 - y2) / 2)
                (x, y) = (res[i][1 + j] + (res[i][3 +j] - res[i][1 + j]) / 2, res[i][0+j] + (res[i][0+j] - res[i][2 +j]) / 2)#((res[i][1 + j] - res[i][0 + j]) ** 2 + (res[i][1 + j] - res[i][3 + j])**2)/2, ((res[i][1 + j] - res[i][0 + j]) ** 2 +(res[i][1 + j] - res[i][3 + j])**2)/2
            #(x, y) = (((i[1 + k] - i[3 + k]) ** 2 + (i[2 + k] - i[3 + k]) ** 2) / 2 * 16), (
                        #((i[1 + k] - i[3 + k]) ** 2 + (i[2 + k] - i[3 + k]) ** 2) / 2 * 16)
            #k += 4
            #z+= 1
            b = np.array([[x], [y]])
            centers.append(np.round(b))
        return centers
