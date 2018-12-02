import numpy as np
import cv2, os, sys
import glob, imutils
from extractHOG import extractHOG


class featureExtractor:
    def __init__(self, subject):
        self.subject = subject
        #roi = cv2.imread('/home/uva-dsa1/Downloads/dip/{}/img/0001.jpg'.format(self.subject))[30:50,65:90]
        roi = cv2.imread("{}/{}/img/0001.jpg".format(os.path.abspath(os.path.dirname(sys.argv[0])), self.subject))[30:50,65:90]
        self.trackSubject(roi)
    #roi = cv2.rectangle(roi,(260,100),(280,120),(255,255,0),2)
    def trackSubject(self, roi):
        hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        cv2.imwrite("template.png",roi)
        centers = []
        for i in range(1,142):
            _file = '/home/uva-dsa1/Downloads/dip/{}/img/%04d.jpg'.format(self.subject)%i

            target = cv2.imread(_file)
            hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
            # calculating object histogram
            roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
            # normalize histogram and apply backprojection
            cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
            dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
            # Now convolute with circular disc
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            cv2.filter2D(dst,-1,disc,dst)
            # threshold and binary AND
            ret,thresh = cv2.threshold(dst,80,255,0)
            thresh = cv2.merge((thresh,thresh,thresh))
            res = cv2.bitwise_and(target,thresh)
            #res = np.vstack((target,thresh,res))
            #thresh = cv2.erode(thresh, None, iterations=1)
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
            cnts = cv2.findContours((thresh).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            prev= [0,0]
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            #print i
            if len(cnts)>0:
                c = max(cnts, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    prev = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
                    centers.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

            else:
                centers.append(prev)
            fragmentCenters = np.array(centers)

            thresh = cv2.rectangle(target,(fragmentCenters[0][0]-15,fragmentCenters[0][1]-15),(fragmentCenters[0][0]+15,fragmentCenters[0][1]+15),(0,255,0),2)

            cv2.imwrite("tracked{}/frame%d.png".format(self.subject)%i, thresh)

        cv2.imwrite("tracked{}/template%d.png".format(self.subject)%i, roi)

fet = featureExtractor('Biker')
