import glob
import cv2
import numpy as np
import time
import pickle
from util import *
import matplotlib.pyplot as plt

# Load images vehicla, non-vehicle
cars = glob.glob("data/vehicle/*.png")
# cars_udacity = glob.glob("data/vehicle_udacity/*.png")
# cars = cars + cars_udacity
notcars = glob.glob("data/non-vehicle/*.png")
# notcars_udacity = glob.glob("data/non-vehicle_udacity/*.png")
# notcars = notcars + notcars_udacity

sample_size = 1
# cars = shuffle(cars, random_state=1)
cars = cars[0:sample_size]
# notcars = shuffle(notcars, random_state=1)
notcars = notcars[0:sample_size]
print('Final size of cars: ', len(cars))
print('Final size of non-cars: ', len(notcars))

###$$$ TRAINING PARAMETERS $$$###
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9 # HOG orientations
pix_per_cell = 8 # HOG pixels per cell 
cell_per_block = 1 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
add_flipped = False # Augment data by flipping

print("Extracting car features...")
# Get features for each set, then stack them together
# Extract color, HOG features


def get_hog_features(image, cspace='RGB', orient=8, pix_per_cell=(8,8), cell_per_block=(2,2), hog_channel=0, vis=False, feat_vec=True):
    if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: 
        feature_image = np.copy(image)
    
    if hog_channel<3:
        img = feature_image[:,:,hog_channel]
    else:
        img = np.copy(feature_image)
    
    #print('feature_image.shape', feature_image.shape)  
    #print('img.shape',img.shape)  
    
    
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=pix_per_cell, cells_per_block=cell_per_block, visualise=vis, transform_sqrt=False, feature_vector=feat_vec, normalise=None)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=pix_per_cell, cells_per_block=cell_per_block, visualise=vis, transform_sqrt=False, feature_vector=feat_vec, normalise=None)
        return features


image = mpimg.imread(cars[0])
# s(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
# car_features = get_hog_features(image[:,:,0], orient, pix_per_cell, cell_per_block, vis=True)

features, hog_ch0 = get_hog_features(image, color_space, orient, 
        (pix_per_cell,pix_per_cell), (cell_per_block,cell_per_block), 0, True)

features, hog_ch1 = get_hog_features(image, color_space, orient, 
        (pix_per_cell,pix_per_cell), (cell_per_block,cell_per_block), 1, True)

features, hog_ch2 = get_hog_features(image, color_space, orient, 
        (pix_per_cell,pix_per_cell), (cell_per_block,cell_per_block), 2, True)

plt.figure(1)
plt.figure(figsize=(15,15))
feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

plt.subplot(431)
plt.title(color_space + "-Ch0")
plt.imshow(feature_image[:,:,0], cmap='gray')
plt.subplot(432)
plt.title(color_space + "Ch1")
plt.imshow(feature_image[:,:,1], cmap='gray')
plt.subplot(433)
plt.title(color_space + "Ch2")
plt.imshow(feature_image[:,:,2], cmap='gray')

plt.subplot(434)
plt.title("Hog-Ch0")
plt.imshow(hog_ch0, cmap='gray')
plt.subplot(435)
plt.title("Hog-Ch1")
plt.imshow(hog_ch1, cmap='gray')
plt.subplot(436)
plt.title("Hog-Ch2")
plt.imshow(hog_ch2, cmap='gray')



image = mpimg.imread(notcars[0])

features, hog_ch0 = get_hog_features(image, color_space, orient, 
        (pix_per_cell,pix_per_cell), (cell_per_block,cell_per_block), 0, True)

features, hog_ch1 = get_hog_features(image, color_space, orient, 
        (pix_per_cell,pix_per_cell), (cell_per_block,cell_per_block), 1, True)

features, hog_ch2 = get_hog_features(image, color_space, orient, 
        (pix_per_cell,pix_per_cell), (cell_per_block,cell_per_block), 2, True)

feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

plt.subplot(437)
plt.title("Not car "+ color_space + "-Ch0")
plt.imshow(feature_image[:,:,0], cmap='gray')
plt.subplot(438)
plt.title("Not car "+ color_space + "Ch1")
plt.imshow(feature_image[:,:,1], cmap='gray')
plt.subplot(439)
plt.title("Not car "+ color_space + "Ch2")
plt.imshow(feature_image[:,:,2], cmap='gray')

plt.subplot(4,3,10)
plt.title("Not car "+ "Hog-Ch0")
plt.imshow(hog_ch0, cmap='gray')
plt.subplot(4,3,11)
plt.title("Not car "+ "Hog-Ch1")
plt.imshow(hog_ch1, cmap='gray')
plt.subplot(4,3,12)
plt.title("Not car "+ "Hog-Ch2")
plt.imshow(hog_ch2, cmap='gray')

plt.savefig('examples/features.png')
