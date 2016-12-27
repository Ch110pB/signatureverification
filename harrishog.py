import numpy as np
import cv2
from cyvlfeat import hog,kmeans

path = 'C:/Users/llop/Documents/Winter 16/Datasets/dataset/'
path_org = 'org/'
path_forg = 'forg/'
num_org = 5
num_forg = 5
hogmatrix = []
hogmatrixcat = []
points = []
patchsize = 10
cellsize = 8
ncells = 1
k = 500 #k for k-means

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
    
    for j in range(npoints):
        roi[0:2] = [np.uint32(harrispoints[0][j]-halfsize),np.uint32(harrispoints[1][j]-halfsize)]
        if all(roi[0:2]) >= 1 and roi[1]+roi[3]-1 <= imsz[0] and roi[0]+roi[2]-1 <= imsz[1]:
            im_tmp = im[(roi[0]-1):(roi[0]+roi[2]),(roi[1]-1):(roi[1]+roi[3])]
            hogi = hog.hog(im_tmp,cellsize,'DalalTriggs')
            validPointCount = validPointCount + 1;
            hogvalues[validPointCount-1,:] = hogi[:]
            validPointIdx[0,validPointCount-1] = j  #store valid indices
            
    hogvalues = hogvalues[0:validPointCount,:]
    validPointIdx = validPointIdx[0:validPointCount]
    for m in range(np.size(points[0][0])):
        valid_points.append([points[0][0][m],points[0][1][m]])
    return([(hogvalues,valid_points)])
    
    
for i in range(1,num_org+1):
    filename = ''.join([path,path_org,'original_',str(i),'_1.png'])
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
    for m in range(hogvalues.shape[0]):
        hogvalues[m] /= sum(hogvalues[m])
    hogmatrix.append(hogvalues)
        
        
    img[dmax]=[0,0,255]
    
#    cv2.imshow('image',img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
hogmatrixcat = np.concatenate((hogmatrix[:]),axis=0)
kmeancenters = kmeans.kmeans(hogmatrixcat,k,initialization='PLUSPLUS')
kmeanclusters = kmeans.kmeans_quantize(hogmatrixcat,kmeancenters)

label_org = []
clustercount = 0

for i in range(len(hogmatrix)):
    count = 0
    label2 = []
    while count < hogmatrix[i].shape[0]:
        label2.append(kmeanclusters[clustercount])
        clustercount += 1
        count += 1
    label_org.append(label2)

hist_org=np.zeros((len(label_org),k))
    
#for i in range(len(label)):
#    for j in range(k):
#        hist[i][j] = label[i].count(j)
        
for i in range(len(label_org)):
    for j in range(len(label_org[i])):
        hist_org[i][label_org[i][j]] += 1
    hist_org[i] /= sum(hist_org[i])

hogmatrix = []
points = []
    
for i in range(1,num_forg+1):
    filename = ''.join([path,path_forg,'forgeries_',str(i),'_1.png'])
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
    for m in range(hogvalues.shape[0]):
        hogvalues[m] /= sum(hogvalues[m])
    hogmatrix.append(hogvalues)
        
        
    img[dmax]=[0,0,255]
#    
#    cv2.imshow('image',img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
hogmatrixcat = np.concatenate((hogmatrix[:]),axis=0)
kmeancenters = kmeans.kmeans(hogmatrixcat,k,initialization='PLUSPLUS')
kmeanclusters = kmeans.kmeans_quantize(hogmatrixcat,kmeancenters)

label_forg = []
clustercount = 0

for i in range(len(hogmatrix)):
    count = 0
    label2 = []
    while count < hogmatrix[i].shape[0]:
        label2.append(kmeanclusters[clustercount])
        clustercount += 1
        count += 1
    label_forg.append(label2)

hist_forg=np.zeros((len(label_forg),k))
        
for i in range(len(label_forg)):
    for j in range(len(label_forg[i])):
        hist_forg[i][label_forg[i][j]] += 1
    hist_forg[i] /= sum(hist_forg[i])
    
#globaldesc=[[(np.zeros(k),0)]*10]*5
##globaldesc = np.zeros((5,10))
#for i in range(5):
#    for j in range(10):
#        globaldesc[i][j]=(abs(hist_org[i]-hist_org[j]),1)
    
#for i in range(2):
#    for j in range(3):
#        print(i)
#        globaldesc[i][j]=(abs(hist_org[i]-hist_org[j]),1)
#        print(j)

#for i in range(num_org):
#    for j in range(num_org):
#        globaldesc[i][j] = (abs(hist_org[i]-hist_org[j]),1)
#for i in range(num_org):
#    for j in range(num_forg):
#        globaldesc[i][j+num_org] = (abs(hist_org[i]-hist_forg[j]),0)