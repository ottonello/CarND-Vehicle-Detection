import glob
import cv2
import numpy as np
import time
import pickle
from util import *
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Load images vehicla, non-vehicle
cars = glob.glob("data/vehicle/*.png")
# cars_udacity = glob.glob("data/vehicle_udacity/*.png")
# cars = cars + cars_udacity
# TOOD add udacity dataset
notcars = glob.glob("data/non-vehicle/*.png")
# notcars_udacity = glob.glob("data/non-vehicle_udacity/*.png")
# notcars = notcars + notcars_udacity

sample_size = 5000
cars = shuffle(cars, random_state=1)
cars = cars[0:sample_size]
notcars = shuffle(notcars, random_state=1)
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
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat, add_flipped=add_flipped)
print("Extracting not-car features...")
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat, add_flipped=add_flipped)
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

print('Final size of car features: ', len(car_features))
print('Final size of not car features: ', len(notcar_features))

print("Normalizing...")
# Normalize features
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

# Label data
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

print("Splitting...")
# Split into training & test
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print("Training...")
print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
clf = LinearSVC()

# Train classifier
# Check the training time for the SVC
t=time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))

# Save classifier
clf_file = "clf.pkl"
print("Saving classifier, scaler and parameters into file: ", clf_file)

dist_pickle = {}
dist_pickle["clf"]= clf
dist_pickle["color_space"]= color_space
dist_pickle["scaler"]= X_scaler
dist_pickle["orient"]= orient
dist_pickle["pix_per_cell"]= pix_per_cell
dist_pickle["cell_per_block"]=cell_per_block
dist_pickle["spatial_size"]=spatial_size
dist_pickle["hist_bins"]= hist_bins
dist_pickle["hog_channel"]= hog_channel
dist_pickle["spatial_feat"]= spatial_feat
dist_pickle["hist_feat"]= hist_feat
dist_pickle["hog_feat"]= hog_feat


pickle.dump(dist_pickle,  open(clf_file, 'wb'))
