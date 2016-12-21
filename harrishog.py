import numpy as np
import cv2

path = 'C:/Users/llop/Documents/Winter 16/Datasets/dataset/'
hogmatrix = []
points=[]

for i in range(1,6):
    filename = ''.join([path,str(i),'.png'])
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    points.append()
    img[dst>0.01*dst.max()]=[255,0,0]
    
    
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    hogmatrix.append(h)
    
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
