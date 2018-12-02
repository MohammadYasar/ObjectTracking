import numpy as np
import pandas as pd
import cv2, time, imutils
from rms import non_max_suppression_slow
from skimage.feature import hog

class extractHOG:
    def __init__(self, image, winSize, blockSize, blockStride, cellSize, nbins, deprivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, winStride, padding):
        self.winSize = winSize
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize
        self.nbins = nbins
        self.deprivAperture = deprivAperture
        self.winSigma = winSigma
        self.histogramNormType = histogramNormType
        self.L2HysThreshold = L2HysThreshold
        self.gammaCorrection = gammaCorrection
        self.nlevels = nlevels
        self.image = image
        self.winStride = winStride
        self.padding = padding
        self.getSlidingWindow(image)
        self.subject = 'Biker'
        #self.getNMS()

    def slidingWindow(self, image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield(x,y, image[y:y+windowSize[1], x:x+windowSize[0]])

    def pyramid(self, image, scale=1.5, minSize=(30, 30)):
    	# yield the original image
    	yield image
    	# keep looping over the pyramid
    	while True:
    		# compute the new dimensions of the image and resize it
    		w = int(image.shape[1] / scale)
    		image = imutils.resize(image, width=w)

    		# if the resized image does not meet the supplied minimum
    		# size, then stop constructing the pyramid
    		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
    			break

    		# yield the next image in the pyramid
    		yield image

    def getSlidingWindow(self, image):
        winW = 32
        winH =32
        x1 = 260
        x2 = 280
        y1 = 95
        y2 = 121
        stepSize = 16
        labels = []
        """
        cv2.rectangle(image, (x1,y1), (x2, y2), (0,255,255),2)
        cv2.waitKey(10000)
        """
        count = 0
        label = 0
        instances = []
        #hog = cv2.HOGDescriptor()
        hog_vector = []
        for resized in self.pyramid(image, scale = 1.5):
            print "{} {} {} {}".format(x1,x2,y1,y2)
            if count >0:
                x1 = int(x1/1.5)
                x2 = int(x2/1.5)
                y1 = int(y1/1.5)
                y2 = int(y2/1.5)
                stepSize = int(stepSize/1.5)
                if stepSize<1:
                    stepSize = 1
            for (x,y,window) in self.slidingWindow(resized, stepSize = stepSize, windowSize = (winW,winH)):
                if window.shape[0]!=winH or window.shape[1]!=winW:
                    continue
                clone  = resized.copy()
                window = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) #bring it donw to sliding windows
                hog_vector.extend(hog(window, orientations=4, pixels_per_cell=(4, 4),cells_per_block=(1, 1), visualise=False))

                if x in range(x1,x2) and y in range(y1,y2):
                    print x
                    labels.append([x,y,x+winW, y+winH, 1])
                    label = 1
                else:
                    labels.append([x,y,x+winW, y+winH, 0])
                    label = 0

                df = pd.DataFrame(data = (labels), columns = ["x1", "y1", "x2","y2", "label"])
                df.to_csv("_file.csv", sep = ',')
                """
                cv2.rectangle(clone, (x,y), (x+winW, y+winH), (0,255,255),2)
                cv2.imshow("window", clone)
                cv2.waitKey(1)
                time.sleep(0.025)
                """
            count = count + 1
    def getNMS(self):
        _file = '/home/uva-dsa1/Downloads/dip/{}/img/%04d.jpg'.format(self.subject)%1
        roi = (100,120,264,275)
        x1 = 264
        x2 = 274
        y1 = 100
        y2 = 120
        images = [('/home/uva-dsa1/Downloads/dip/{}/img/%04d.jpg'.format(self.subject)%1, np.array([(x1, y1, x2, y2),(x1-5, y1-5, x2-5, y2-5),(x1+5, y1+5, x2+5, y2+5),((x1-3, y1-3, x2-3, y2-3)),((x1+3, y1+3, x2+3, y2+3)),	((x1, y1, x2, y2))]))]
        for (imagePath, boundingBoxes) in images:
        	# load the image and clone it
        	print "[x] %d initial bounding boxes" % (len(boundingBoxes))
        	image = cv2.imread(imagePath)
        	orig = image.copy()

        	# loop over the bounding boxes for each image and draw them
        	for (startX, startY, endX, endY) in boundingBoxes:
        		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

        	# perform non-maximum suppression on the bounding boxes
        	pick = non_max_suppression_slow(boundingBoxes, 0.3)
        	print "[x] after applying non-maximum, %d bounding boxes" % (len(pick))

        	# loop over the picked bounding boxes and draw them
        	for (startX, startY, endX, endY) in pick:
        		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        	# display the images
        	cv2.imshow("Original", orig)
        	cv2.waitKey(1000)
        	cv2.imshow("After NMS", image)
        	cv2.waitKey(10000)
