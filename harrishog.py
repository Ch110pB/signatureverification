import numpy as np
from scipy.spatial import Delaunay
import cv2
from cyvlfeat import hog,kmeans

path = 'C:/Users/llop/Documents/Winter 16/Datasets/dataset/'
#path = 'D:/IMPORTANT/signature_verification/Datasets/dataset/'
path_org = 'org/'
path_forg = 'forg/'
num_org = 5
num_forg = 5
hogmatrix = []
hogmatrixcat = []
points_org = []
points_forg = []
patchsize = 10
cellsize = 8
ncells = 1
kvalue = 500 #k for k-means

def calcHog(im,points,patchsize,ncells,idx):
    imsz = im.shape
    halfsize = (np.float32(patchsize) - np.float32(patchsize%2))/2
    valid_points = []
    
    npoints = len(points[idx][0])
    roi = [1,1,patchsize,patchsize]
    
    validPointIdx = np.zeros((1,npoints),np.uint32)
    validPointCount = np.zeros((1),np.uint32)
    
    hogdim = [npoints,36*ncells**2]
    hogvalues = np.zeros(hogdim,np.float32)
    
    for j in range(npoints):
        roi[0:2] = [np.uint32(points[idx][0][j]-halfsize),np.uint32(points[idx][1][j]-halfsize)]
        if all(roi[0:2]) >= 1 and roi[1]+roi[3]-1 <= imsz[0] and roi[0]+roi[2]-1 <= imsz[1]:
            im_tmp = im[(roi[0]-1):(roi[0]+roi[2]),(roi[1]-1):(roi[1]+roi[3])]
            hogi = hog.hog(im_tmp,cellsize,'DalalTriggs')
            validPointCount = validPointCount + 1;
            hogvalues[validPointCount-1,:] = hogi[:]
            validPointIdx[0,validPointCount-1] = j  #store valid indices
            
    hogvalues = hogvalues[0:validPointCount,:]
    validPointIdx = validPointIdx[0:validPointCount]
    for m in range(np.size(points[idx][0])):
        valid_points.append([points[idx][0][m],points[idx][1][m]])
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
    points_org.append(harrispoints)
    
    [(hogvalues,valid_points)] = calcHog(gray,points_org,patchsize,ncells,i-1)
    for m in range(hogvalues.shape[0]):
        hogvalues[m] /= sum(hogvalues[m])
    hogmatrix.append(hogvalues)
        
        
    img[dmax]=[0,0,255]
    
#    cv2.imshow('image',img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
hogmatrixcat = np.concatenate((hogmatrix[:]),axis=0)
kmeancenters = kmeans.kmeans(hogmatrixcat,kvalue,initialization='PLUSPLUS')
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

hist_org=np.zeros((len(label_org),kvalue))
    
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
    points_forg.append(harrispoints)
    
    [(hogvalues,valid_points)] = calcHog(gray,points_forg,patchsize,ncells,i-1)
    for m in range(hogvalues.shape[0]):
        hogvalues[m] /= sum(hogvalues[m])
    hogmatrix.append(hogvalues)
        
        
    img[dmax]=[0,0,255]
#    
#    cv2.imshow('image',img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
hogmatrixcat = np.concatenate((hogmatrix[:]),axis=0)
kmeancenters = kmeans.kmeans(hogmatrixcat,kvalue,initialization='PLUSPLUS')
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

hist_forg=np.zeros((len(label_forg),kvalue))
        
for i in range(len(label_forg)):
    for j in range(len(label_forg[i])):
        hist_forg[i][label_forg[i][j]] += 1
    hist_forg[i] /= sum(hist_forg[i])
    
globaldesc_temp = []
globaldesc = []
count = 0
for i in range(num_org):
    globaldesc_org = []
    globaldesc_forg = []
    for j in range(num_org):
        globaldesc_org.append((abs(hist_org[i]-hist_org[j]),1))
    globaldesc_temp.append(globaldesc_org)
    for k in range(num_forg):
        globaldesc_forg.append((abs(hist_org[i]-hist_forg[k]),0))
    globaldesc_temp.append(globaldesc_forg)
for i in range(num_org):
    globaldesc.append(globaldesc_temp[count]+globaldesc_temp[count+1])
    count += 2
    
tri_org_temp = []
tri_org = []
tri_org_center = []
hogmatrix_tri_org = []
kmeancenters_tri_org = []
kmeanclusters_tri_org = []
for i in range(num_org):
    temp = []
    tri_org_center_temp_1 = []
    tri_org_center_temp_2 = []
    for j in range(len(points_org[i][0])):
        temp.append((points_org[i][0][j],points_org[i][1][j]))
    tri_org_temp = Delaunay(temp)
    tri_org.append(tri_org_temp.points[tri_org_temp.simplices])
    for k in range(tri_org_temp.simplices.shape[0]):
        tri_org_center_temp_1.append((sum(tri_org[i][k])/3)[0])
        tri_org_center_temp_2.append((sum(tri_org[i][k])/3)[1])
    tri_org_center.append([np.array(tri_org_center_temp_1),np.array(tri_org_center_temp_2)])
    [(hogvalues_tri_org,p)] = calcHog(gray,tri_org_center,patchsize,ncells,i)
    for n in range(hogvalues_tri_org.shape[0]):
        hogvalues_tri_org[n] /= sum(hogvalues_tri_org[n])
    hogmatrix_tri_org.append(hogvalues_tri_org)
    
hogmatrixcat_tri_org = np.concatenate((hogmatrix_tri_org[:]),axis=0)
kmeancenters_tri_org = kmeans.kmeans(hogmatrixcat_tri_org,kvalue,initialization='PLUSPLUS')
kmeanclusters_tri_org = kmeans.kmeans_quantize(hogmatrixcat_tri_org,kmeancenters_tri_org)

tri_forg_temp = []
tri_forg = []
tri_forg_center = []
hogmatrix_tri_forg = []
kmeancenters_tri_forg = []
kmeanclusters_tri_forg = []
for i in range(num_forg):
    temp = []
    tri_forg_center_temp_1 = []
    tri_forg_center_temp_2 = []
    for j in range(len(points_forg[i][0])):
        temp.append((points_forg[i][0][j],points_forg[i][1][j]))
    tri_forg_temp = Delaunay(temp)
    tri_forg.append(tri_forg_temp.points[tri_forg_temp.simplices])
    for k in range(tri_forg_temp.simplices.shape[0]):
        tri_forg_center_temp_1.append((sum(tri_forg[i][k])/3)[0])
        tri_forg_center_temp_2.append((sum(tri_forg[i][k])/3)[1])
    tri_forg_center.append([np.array(tri_forg_center_temp_1),np.array(tri_forg_center_temp_2)])
    [(hogvalues_tri_forg,p)] = calcHog(gray,tri_forg_center,patchsize,ncells,i)
    for m in range(hogvalues_tri_forg.shape[0]):
        hogvalues_tri_forg[m] /= sum(hogvalues_tri_forg[m])
    hogmatrix_tri_forg.append(hogvalues_tri_forg)
    
hogmatrixcat_tri_forg = np.concatenate((hogmatrix_tri_forg[:]),axis=0)
kmeancenters_tri_forg = kmeans.kmeans(hogmatrixcat_tri_forg,kvalue,initialization='PLUSPLUS')
kmeanclusters_tri_forg = kmeans.kmeans_quantize(hogmatrixcat_tri_forg,kmeancenters_tri_forg)
    
label_tri_org = []
clustercount_tri = 0

for i in range(len(hogmatrix_tri_org)):
    count = 0
    label2 = []
    while count < hogmatrix_tri_org[i].shape[0]:
        label2.append(kmeanclusters_tri_org[clustercount_tri])
        clustercount_tri += 1
        count += 1
    label_tri_org.append(label2)

hist_tri_org=np.zeros((len(label_tri_org),kvalue))
    
for i in range(len(label_tri_org)):
    for j in range(len(label_tri_org[i])):
        hist_tri_org[i][label_tri_org[i][j]] += 1
    hist_tri_org[i] /= sum(hist_tri_org[i])

label_tri_forg = []
clustercount_tri = 0

for i in range(len(hogmatrix_tri_forg)):
    count = 0
    label2 = []
    while count < hogmatrix_tri_forg[i].shape[0]:
        label2.append(kmeanclusters_tri_forg[clustercount_tri])
        clustercount_tri += 1
        count += 1
    label_tri_forg.append(label2)

hist_tri_forg=np.zeros((len(label_tri_forg),kvalue))
    
for i in range(len(label_tri_forg)):
    for j in range(len(label_tri_forg[i])):
        hist_tri_forg[i][label_tri_forg[i][j]] += 1
    hist_tri_forg[i] /= sum(hist_tri_forg[i])

globaldesc_tri_temp = []
globaldesc_tri = []
count_tri = 0
for i in range(num_org):
    globaldesc_tri_org= []
    globaldesc_tri_forg = []
    for j in range(num_org):
        globaldesc_tri_org.append((abs(hist_tri_org[i]-hist_tri_org[j]),1))
    globaldesc_tri_temp.append(globaldesc_tri_org)
    for k in range(num_forg):
        globaldesc_tri_forg.append((abs(hist_tri_org[i]-hist_tri_forg[k]),0))
    globaldesc_tri_temp.append(globaldesc_tri_forg)
for i in range(num_org):
    globaldesc_tri.append(globaldesc_tri_temp[count_tri]+globaldesc_tri_temp[count_tri+1])
    count_tri += 2
    
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