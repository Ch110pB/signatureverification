import numpy as np
import cv2
from cyvlfeat import hog,kmeans

path = 'C:/Users/llop/Documents/Winter 16/Datasets/dataset/'
hogmatrix = []
hogmatrixcat = []
points = []
patchsize = 10
cellsize = 8
ncells = 1

def calcHog(im,points,patchsize,ncells):
    imsz = im.shape
    halfsize = (np.float32(patchsize) - np.float32(patchsize%2))/2
    valid_points = []
    
    npoints = len(harrispoints[0])
    roi = [1,1,patchsize,patchsize]
    
    validPointIdx = np.zeros((1,npoints),np.uint32)
    validPointCount = np.zeros((1),np.uint32)
    
    hogdim = [npoints,36*ncells**2]
    hogvalues = np.zeros(hogdim,np.float32)
    
    for j in range(0,npoints):
        roi[0:2] = [np.uint32(harrispoints[0][j]-halfsize),np.uint32(harrispoints[1][j]-halfsize)]
        if all(roi[0:2]) >= 1 and roi[1]+roi[3]-1 <= imsz[0] and roi[0]+roi[2]-1 <= imsz[1]:
            im_tmp = im[(roi[0]-1):(roi[0]+roi[2]),(roi[1]-1):(roi[1]+roi[3])]
            hogi = hog.hog(im_tmp,cellsize,'DalalTriggs')
            validPointCount = validPointCount + 1;
            hogvalues[validPointCount-1,:] = hogi[:]
            validPointIdx[0,validPointCount-1] = i  #store valid indices
            
    hogvalues = hogvalues[0:validPointCount+1,:]
    validPointIdx = validPointIdx[0:validPointCount+1]
    for k in range(0,np.size(points[0][0])):
        valid_points.append([points[0][0][k],points[0][1][k]])
    return([(hogvalues,valid_points)])


for i in range(1,6):
    filename = ''.join([path,str(i),'.png'])
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    #threshold for an optimal value, it may vary depending on the image.
    dmax = dst>0.01*dst.max()
    harrispoints = np.where(dmax == True)
    points.append(harrispoints)
    
    [(hogvalues,valid_points)] = calcHog(gray,points,patchsize,ncells)
    hogmatrix.append(hogvalues)
    
    img[dmax]=[0,0,255]
        
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
hogmatrixcat = np.concatenate((hogmatrix[0:5]),axis=0)
kmeancenters = kmeans.kmeans(hogmatrixcat,4)
kmeanclusters = kmeans.kmeans_quantize(hogmatrixcat,kmeancenters)