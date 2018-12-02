import numpy as np
import cv2

class extractHOG:
    def __init__(self, image, winSize, blockSize, blockStride, cellSize, nbins, deprivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels):
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
        self.getHogFeatures()
    def getHogFeatures(self):
        locations = ((10,20),)
        hog = cv2.HOGDescriptor(self.winSize, self.blockSize, self.blockStride, self.cellSize, self.nbins, self.deprivAperture, self.histogramNormType, self.L2HysThreshold, self.gammaCorrection, self.nlevels)
        hist = hog.compute(self.image, self.winStride, self.padding, locations)
        
