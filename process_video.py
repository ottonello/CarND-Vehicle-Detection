from moviepy.editor import VideoFileClip
import pickle
from pipeline import find_cars

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
scale = 2.0


def process_image(img):
    return find_cars(img, 
		ystart, ystop, scale, clf, color_space, X_scaler, orient,
		 pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel,
		 spatial_feat, hist_feat, hog_feat)


output_video = "solution_video.mp4"
clip1 = VideoFileClip("project_video.mp4")
output_clip= clip1.fl_image(process_image)
output_clip.write_videofile(output_video, audio=False)
