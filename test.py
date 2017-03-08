import os
import pickle
import cv2
import matplotlib.image as mpimg
import glob
from pipeline import *
from matplotlib import pyplot as plt


# Load classifier and parameters
dist_pickle = pickle.load( open("clf.pkl", "rb" ) )
clf = dist_pickle["clf"] # Classifier
color_space = dist_pickle["color_space"] # Color space in which to process the image(input is RGB)
X_scaler = dist_pickle["scaler"] # Scaler used to normalize features

spatial_size = dist_pickle["spatial_size"] # Size of spatial features
hist_bins = dist_pickle["hist_bins"] # Number of bins to take in color histogram

# HOG Options
orient = dist_pickle["orient"] # Orientation bins
pix_per_cell = dist_pickle["pix_per_cell"] # Pixels per cell
cell_per_block = dist_pickle["cell_per_block"] # Cells per block
hog_channel = dist_pickle["hog_channel"] # Number of channel to take HOG(or 'ALL')

spatial_feat = dist_pickle["spatial_feat"] # Spatial features on or off
hist_feat = dist_pickle["hist_feat"] # Histogram features on or off
hog_feat = dist_pickle["hog_feat"] # HOG features on or off

ystart = 336
ystop = 656
scale = 1.5

for filename in glob.glob("test_images/*.jpg"):
	print(filename)
	img = mpimg.imread(filename)

	out_img = find_cars(img, 
		ystart, ystop, scale, clf, color_space, X_scaler, orient,
		 pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel,
		 spatial_feat, hist_feat, hog_feat, False)

	plt.imshow(out_img)
	orig_filename, _ = os.path.splitext(filename)
	orig_filename = orig_filename.replace('test_images/', '')

	plt.savefig('output_images/' + orig_filename + '.png')
