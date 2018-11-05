import sys
import cv2
import os
import csv
import dlib
import imutils
import argparse
import numpy as np
import math
from PyQt5.QtWidgets import (QMainWindow, QApplication, QFontDialog, QStyleFactory, QAction, QMessageBox, QFileDialog)
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.uic import loadUi


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('main.ui', self)

        self.webcamStartBtn.clicked.connect(self.start_webcam)
        self.webcamStopBtn.clicked.connect(self.stop_webcam)
        self.eyesDetectBtn.setCheckable(True)
        self.eyesDetectBtn.toggled.connect(self.detect_webcam_eye)
        self.motionDetectBtn.setCheckable(True)
        self.motionDetectBtn.toggled.connect(self.detect_webcam_motion)
        self.motimgButton.clicked.connect(self.set_motion_image)
        self.faceAverageBtn.clicked.connect(self.set_face_average_image)
        self.realTimeFaceAverageBtn.clicked.connect(self.set_real_time_average_image)
        # Test for real time face average to start webcam
        #self.realTimeFaceAverageBtn.clicked.connect(self.start_webcam)
        self.faceSwapLoad1.clicked.connect(lambda: self.loadClicked(1))
        self.faceSwapLoad2.clicked.connect(lambda: self.loadClicked2(1))
        self.faceSwapBtn.clicked.connect(self.face_swap_start)
        self.faceMorphLoad1.clicked.connect(lambda: self.loadClicked(2))
        self.faceMorphLoad2.clicked.connect(lambda: self.loadClicked2(2))
        self.faceMorphBtn.clicked.connect(self.perform_face_morph)

        self.image = None
        self.face_enabled = False
        self.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.motion_enabled = False
        self.motionFrame = None
        self.face_average_enabled = False
        self.real_time_face_average_enabled = False
        self.path1 = None
        self.path2 = None


        quitAction = QAction('&Quit', self)
        quitAction.setShortcut("Ctrl+Q")
        quitAction.setStatusTip("Leave the app")
        quitAction.triggered.connect(self.close_application)

        helpAction = QAction("&Help", self)
        helpAction.setShortcut("Ctrl+H")
        helpAction.setStatusTip("Help")
        helpAction.triggered.connect(self.get_help)

        self.statusBar()

        mainMenu = self.menuBar()

        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(quitAction)

        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction(helpAction)

    @pyqtSlot()
    # n here is to pass button information between face swap and face morphing
    def loadClicked(self, n):
        fname,_ = QFileDialog.getOpenFileName(self, 'Open File' +
                                              "Image Files", '.',
                                              "(*.png *.jpg *.jpeg)")
        if fname:
            if n == 1:
                self.loadImage(fname, 1)
            elif n == 2:
                self.loadImage(fname, 3)
        else:
            print('Invalid file extension')

    # For face swap's second label
    def loadClicked2(self, n):
        fname,_ = QFileDialog.getOpenFileName(self, 'Open File' +
                                              "Image Files", '.',
                                              "(*.png *.jpg *.jpeg)")
        if fname:
            if n == 1:
                self.loadImage(fname, 2)
            elif n == 2:
                self.loadImage(fname, 4)
        else:
            print('Invalid file extension')

    def loadImage(self, fname, my_path=0):
        if my_path == 1:
            self.path1 = fname
            self.image = cv2.imread(fname)
            self.displayImage(1)
        elif my_path == 2:
            self.path2 = fname
            self.image = cv2.imread(fname)
            self.displayImage(2)
        elif my_path == 3:
            self.path1 = fname
            self.image = cv2.imread(fname)
            self.displayImage(3)
        elif my_path == 4:
            self.path2 = fname
            self.image = cv2.imread(fname)
            self.displayImage(4)
        else:
            print('FACE SWAP ERROR PATH')

    def displayImage(self, my_path):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:  # rows[0], cols[1], channels[2]
            if(self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0],
                     self.image.strides[0], qformat)
        # BGR ==> RGB
        img = img.rgbSwapped()
        if my_path == 1:
            self.faceSwapLabel1.setPixmap(QPixmap.fromImage(img))
            self.faceSwapLabel1.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.faceSwapLabel1.setScaledContents(True)
        elif my_path == 2:
            self.faceSwapLabel2.setPixmap(QPixmap.fromImage(img))
            self.faceSwapLabel2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.faceSwapLabel2.setScaledContents(True)
        elif my_path == 3:
            self.faceMorphLabel1.setPixmap(QPixmap.fromImage(img))
            self.faceMorphLabel1.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.faceMorphLabel1.setScaledContents(True)
        elif my_path == 4:
            self.faceMorphLabel2.setPixmap(QPixmap.fromImage(img))
            self.faceMorphLabel2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.faceMorphLabel2.setScaledContents(True)

    # Till here the functions are not in use ---------------------

    # Functions are in use from now on:
    def get_help(self):
        # Fill here later
        pass

    def detect_webcam_motion(self, status):
        if status:
            self.motion_enabled = True
            self.motionDetectBtn.setText('Detection Stop')
        else:
            self.motion_enabled = False
            self.motionDetectBtn.setText('Motion Detection')

    def set_motion_image(self):
        gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        self.motionFrame = gray
        self.displayImage2(self.motionFrame, 2)

    # Face average
    def set_face_average_image(self):
        average_img = self.start_face_average()
        self.displayImage2(average_img, 3)

    # Real time face average
    def set_real_time_average_image(self):
        real_time_average_img = self.start_real_time_average()
        self.displayImage2(real_time_average_img, 4)

    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)

        if self.face_enabled:
            detected_image = self.detect_eye(self.image)
            self.displayImage2(detected_image, 1)
        elif self.motion_enabled:
            detected_motion = self.detect_motion(self.image.copy())
            self.displayImage2(detected_motion, 1)
        else:
            self.displayImage2(self.image, 1)

    def detect_motion(self, input_img):
        self.text = 'No Motion'
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frameDiff = cv2.absdiff(self.motionFrame, gray)
        thresh = cv2.threshold(frameDiff, 40, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=5)

        im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        try:
            hierarchy = hierarchy[0]
        except:
            hierarchy = []

        height, width, channels = input_img.shape
        min_x, min_y = width, height
        max_x = max_y = 0

        for contour, hier in zip(cnts, hierarchy):
            (x, y, w, h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x+w, max_x)
            min_y, max_y = min(y, min_y), max(y+h, max_y)

        if max_x - min_x > 80 and max_y - min_y > 80:
            cv2.rectangle(input_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
            self.text = 'Motion Detected'

        cv2.putText(input_img, 'Motion Status: {}'.format(self.text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

        return input_img

    def detect_eye(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(90, 90))

        for x,y,w,h in faces:
            if self.faceCheckBox.isChecked():
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            elif self.eyesCheckBox.isChecked():
                ex, ey, ewidth, eheight = int(x + 0.125 * w), int(y + 0.25 * h), int(0.75 * w), int(
                    0.25 * h)

                cv2.rectangle(img, (ex, ey), (ex + ewidth, ey + eheight), (128, 255, 0), 2)

        return img

    def stop_webcam(self):
        self.timer.stop()

    def displayImage2(self, img, myWindow=1 ):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # [0]=rows, [1]=cols, [2]=channels
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR ==> RGB
        outImage = outImage.rgbSwapped()

        if myWindow == 1:
            self.videoFeedLabel.setPixmap(QPixmap.fromImage(outImage))
            self.videoFeedLabel.setScaledContents(True)
        if myWindow == 2:
            self.inputImgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.inputImgLabel.setScaledContents(True)
        # For face average
        if myWindow == 3:
            self.faceAverageLbl2.setPixmap(QPixmap.fromImage(outImage))
            #self.faceAverageLbl2.setScaledContents(True)   !Scaling reduces the quality!
        # For face swap
        if myWindow == 4:
            self.faceSwapOutputLabel.setPixmap(QPixmap.fromImage(outImage))
            self.faceAverageLbl2.setScaledContents(True)
        # For face morphing
        if myWindow == 5:
            self.faceMorphOutputLabel.setPixmap(QPixmap.fromImage(outImage))
            self.faceMorphOutputLabel.setScaledContents(True)

    def detect_webcam_eye(self, status):
        if status:
            self.eyesDetectBtn.setText('Stop Detection')
            self.face_enabled = True
        else:
            self.eyesDetectBtn.setText('Detect')
            self.face_enabled = False

    def close_application(self):
        choice = QMessageBox.question(self, 'Confirm Exit',
                                      "Are you sure you want to exit ?",
                                      QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            print("Extracting Now")
            sys.exit()
        else:
            pass

    # Face Average functions --------------------
    def readPoints(self, path):
        pointsArray = []
        for filePath in sorted(os.listdir(path)):
            if filePath.endswith(".txt"):
                points = []
                with open(os.path.join(path, filePath)) as file:
                    for line in file:
                        x, y = line.split(',')
                        points.append((int(x), int(y)))
                file.close()
                pointsArray.append(points)
        return pointsArray

    def readImages(self, path):
        imagesArray = []
        for filePath in sorted(os.listdir(path)):
            if filePath.endswith(".jpg"):
                img = cv2.imread(os.path.join(path, filePath))
                img = np.float32(img) / 255.0
                imagesArray.append(img)
        return imagesArray

    def similarityTransform(self, inPoints, outPoints):
        s60 = math.sin(60 * math.pi / 180)
        c60 = math.cos(60 * math.pi / 180)

        inPts = np.copy(inPoints).tolist()
        outPts = np.copy(outPoints).tolist()

        xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0]
        yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1]

        inPts.append([np.int(xin), np.int(yin)])

        xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0]
        yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1]

        outPts.append([np.int(xout), np.int(yout)])

        tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)

        return tform

    def rectContains(self, rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True

    def calculateDelaunayTriangles(self, rect, points):
        # Create subdiv
        subdiv = cv2.Subdiv2D(rect)

        # Insert points into subdiv
        for p in points:
            subdiv.insert((p[0], p[1]))

        # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
        triangleList = subdiv.getTriangleList()

        # Find the indices of triangles in the points array

        delaunayTri = []

        for t in triangleList:
            pt = []
            pt.append((t[0], t[1]))
            pt.append((t[2], t[3]))
            pt.append((t[4], t[5]))

            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            if self.rectContains(rect, pt1) and self.rectContains(rect, pt2) and self.rectContains(rect, pt3):
                ind = []
                for j in range(3):
                    for k in range(int(len(points))):
                        if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                            ind.append(k)
                if len(ind) == 3:
                    delaunayTri.append((ind[0], ind[1], ind[2]))

        return delaunayTri

    def constrainPoint(self, p, w, h):
        p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
        return p

    def applyAffineTransform(self, src, srcTri, dstTri, size):

        # Given a pair of triangles, find the affine transform.
        warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

        # Apply the Affine Transform just found to the src image
        dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)

        return dst

    def warpTriangle(self, img1, img2, t1, t2):

        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        # Offset points by left top corner of the respective rectangles
        t1Rect = []
        t2Rect = []
        t2RectInt = []

        for i in range(3):
            t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        # Get mask by filling triangle
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

        # Apply warpImage to small rectangular patches
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        size = (r2[2], r2[3])

        img2Rect = self.applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

        img2Rect = img2Rect * mask

        # Copy triangular region of the rectangular patch to the output image
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                    (1.0, 1.0, 1.0) - mask)

        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect

    def start_face_average(self):
        path = 'Face_Averaging/img/'
        import landmarks_extraction

        # Dimensions of output image
        w = 600
        h = 600

        # Read points for all images
        allPoints = self.readPoints(path)

        # Read all images
        images = self.readImages(path)

        # Eye corners
        eyecornerDst = [(np.int(0.3 * w), np.int(h / 3)), (np.int(0.7 * w), np.int(h / 3))]

        imagesNorm = []
        pointsNorm = []

        # Add boundary points for delaunay triangulation
        boundaryPts = np.array(
            [(0, 0), (w / 2, 0), (w - 1, 0), (w - 1, h / 2), (w - 1, h - 1), (w / 2, h - 1), (0, h - 1), (0, h / 2)])

        # Initialize location of average points to 0s
        pointsAvg = np.array([(0, 0)] * (len(allPoints[0]) + len(boundaryPts)), np.float32())
        n = len(allPoints[0])

        numImages = len(images)

        # Warp images and trasnform landmarks to output coordinate system,
        # and find average of transformed landmarks.

        for i in range(numImages):
            points1 = allPoints[i]

            # Corners of the eye in input image
            eyecornerSrc = [allPoints[i][36], allPoints[i][45]]

            # Compute similarity transform
            tform = self.similarityTransform(eyecornerSrc, eyecornerDst)

            # Apply similarity transformation
            img = cv2.warpAffine(images[i], tform, (w, h))

            # Apply similarity transform on points
            points2 = np.reshape(np.array(points1), (68, 1, 2))

            points = cv2.transform(points2, tform)

            points = np.float32(np.reshape(points, (68, 2)))

            # Append boundary points. Will be used in Delaunay Triangulation
            points = np.append(points, boundaryPts, axis=0)

            # Calculate location of average landmark points.
            pointsAvg = pointsAvg + points / numImages

            pointsNorm.append(points)
            imagesNorm.append(img)

        # Delaunay triangulation
        rect = (0, 0, w, h)
        dt = self.calculateDelaunayTriangles(rect, np.array(pointsAvg))

        # Output image
        output = np.zeros((h, w, 3), np.float32())

        # Warp input images to average image landmarks
        for i in range(int(len(imagesNorm))):
            img = np.zeros((h, w, 3), np.float32())
            # Transform triangles one by one
            for j in range(int(len(dt))):
                tin = []
                tout = []

                for k in range(3):
                    pIn = pointsNorm[i][dt[j][k]]
                    pIn = self.constrainPoint(pIn, w, h)

                    pOut = pointsAvg[dt[j][k]]
                    pOut = self.constrainPoint(pOut, w, h)

                    tin.append(pIn)
                    tout.append(pOut)

                self.warpTriangle(imagesNorm[i], img, tin, tout)

            # Add image intensities for averaging
            output = output + img

        # Divide by numImages to get average
        output = output / numImages
        output = (output * 255).round().astype(np.uint8)  # convert numpy float32 to uint8

        return output
    # End of Face Average --------------

    # Real time face average fill it later

    # Face Swap----------------------
    def readPoints_faceSwap(self, path):
        # Create an array of points.
        points = []

        # Read points
        with open(path) as file:
            for line in file:
                x, y = line.split(',')
                points.append((int(x), int(y)))

        return points

    def face_swap_start(self):

        filename1 = os.path.basename(self.path1)
        filename2 = os.path.basename(self.path2)

        img1 = cv2.imread(filename1)
        img2 = cv2.imread(filename2)
        img1Warped = np.copy(img2)

        # Read array of corresponding points
        points1 = self.readPoints_faceSwap(filename1 + '.txt')
        points2 = self.readPoints_faceSwap(filename2 + '.txt')

        # Find convex hull
        hull1 = []
        hull2 = []

        hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

        for i in range(0, len(hullIndex)):
            hull1.append(points1[int(hullIndex[i])])
            hull2.append(points2[int(hullIndex[i])])

        # Find delaunay traingulation for convex hull points
        sizeImg2 = img2.shape
        rect = (0, 0, sizeImg2[1], sizeImg2[0])

        dt = self.calculateDelaunayTriangles(rect, hull2)

        if len(dt) == 0:
            quit()

        # Apply affine transformation to Delaunay triangles
        for i in range(0, len(dt)):
            t1 = []
            t2 = []

            # get points for img1, img2 corresponding to the triangles
            for j in range(0, 3):
                t1.append(hull1[dt[i][j]])
                t2.append(hull2[dt[i][j]])

            self.warpTriangle(img1, img1Warped, t1, t2)

        # Calculate Mask
        hull8U = []
        for i in range(0, len(hull2)):
            hull8U.append((hull2[i][0], hull2[i][1]))

        mask = np.zeros(img2.shape, dtype=img2.dtype)

        cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

        r = cv2.boundingRect(np.float32([hull2]))

        center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

        # Clone seamlessly.
        output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

        self.displayImage2(output, 4)
    # End of face swap------------------

    # Face Morphing-----------------------

    # Warps and alpha blends triangular regions from img1 and img2 to morphing_img
    def morphTriangle(self, img1, img2, img, t1, t2, t, alpha):

        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        r = cv2.boundingRect(np.float32([t]))

        # Offset points by left top corner of the respective rectangles
        t1Rect = []
        t2Rect = []
        tRect = []

        for i in range(3):
            tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
            t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        # Get mask by filling triangle
        mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

        # Apply warpImage to small rectangular patches
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

        size = (r[2], r[3])
        warpImage1 = self.applyAffineTransform(img1Rect, t1Rect, tRect, size)
        warpImage2 = self.applyAffineTransform(img2Rect, t2Rect, tRect, size)

        # Alpha blend rectangular patches
        imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

        # Copy triangular region of the rectangular patch to the output image
        img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask

    def face_morphing_start(self, img1, img2):
        filename1 = os.path.basename(self.path1)
        filename2 = os.path.basename(self.path2)

        #img1 = cv2.imread(filename1)
        #img2 = cv2.imread(filename2)

        alpha = np.arange(0, 1, 0.05)
        # Convert Mat to float data type
        img1 = np.float32(img1)
        img2 = np.float32(img2)
        morph = np.zeros((len(alpha), img1.shape[0], img1.shape[1], img1.shape[2]), dtype=np.uint8)
        for j in range(len(alpha)):

            # Read array of corresponding points
            points1 = self.readPoints_faceSwap(filename1 + '.txt')
            points2 = self.readPoints_faceSwap(filename2 + '.txt')
            points = []
            alpha1 = alpha[j]
            # Compute weighted average point coordinates
            for i in range(int(len(points1))):
                x = (1 - alpha1) * points1[i][0] + alpha1 * points2[i][0]
                y = (1 - alpha1) * points1[i][1] + alpha1 * points2[i][1]
                points.append((x, y))

            # Allocate space for final output
            imgMorph = np.zeros(img1.shape, dtype=img1.dtype)

            # Read triangles from tri.txt
            with open("project_gui/morphing_img/triangulation.txt") as file:
                for line in file:
                    x, y, z = line.split(',')

                    x = int(x)
                    y = int(y)
                    z = int(z)

                    t1 = [points1[x], points1[y], points1[z]]
                    t2 = [points2[x], points2[y], points2[z]]
                    t = [points[x], points[y], points[z]]

                    # Morph one triangle at a time.
                    self.morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha1)

            morph[j] = np.uint8(imgMorph)
        return morph

    def perform_face_morph(self):
        filename1 = os.path.basename(self.path1)
        filename2 = os.path.basename(self.path2)
        img1 = cv2.imread(filename1)
        img2 = cv2.imread(filename2)
        outputs = self.face_morphing_start(img1, img2)
        for output in outputs:
            self.displayImage2(output, 5)
            cv2.imshow('output', output)
            cv2.waitKey(200)

app = QApplication(sys.argv)
window = MainWindow()
window.setWindowTitle('Face Swapper')   # To name the window
window.setStyle(QStyleFactory.create('Fusion'))     # To set the application style
window.setStyleSheet("""
                .MainWindow {
                    background-image: url("app-background.png");
                    }
                """)
window.setWindowIcon(QIcon('python.png'))
window.show()
sys.exit(app.exec_())
