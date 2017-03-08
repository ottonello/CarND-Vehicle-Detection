import numpy as np
import math
from util import *
from scipy.ndimage.measurements import label

# Perform sliding window search on image

# On each window, run extract_features

# Normalize features

# Run classifier on feature

# Generate heatmap from detected windows

# Get labels from heatmap (labels = label(heatmap))
ystart = 336
ystop = 656

def find_cars(img, svc, color_space, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
     hog_channel='ALL', spatial_feat=True, hist_feat=True, hog_feat=True, do_threshold=True):
    draw_img = np.copy(img)

    cells_per_step1 = 1.5  # Instead of overlap, define how many cells to step
    scale1 = 1
    box_list_1 = do_window_search(img, scale1, cells_per_step1, svc, color_space, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, 
     hog_channel, spatial_feat, hist_feat, hog_feat, do_threshold)

    cells_per_step2 = 1
    scale2 = 1.5
    box_list_2 = do_window_search(img, scale2, cells_per_step2, svc, color_space, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
     hog_channel, spatial_feat, hist_feat, hog_feat, do_threshold)

    box_list = np.concatenate([x for x in [box_list_1, box_list_2] if len(x) > 0]) if len(box_list_1) or len(box_list_2) else []

    if do_threshold:
        heat = np.zeros_like(img[:,:,0]).astype(np.float)    
        # Add heat to each box in box list
        heat = add_heat(heat,box_list)
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,1)
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        draw_img = draw_labeled_bboxes(draw_img, labels)
    if not do_threshold:
        for box in box_list:
            cv2.rectangle(draw_img,(box[0][0], box[0][1]),(box[1][0],box[1][1]),(0,0,255),6) 

    return draw_img

def do_window_search(img, scale, cells_per_step, svc, color_space, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
     hog_channel='ALL', spatial_feat=True, hist_feat=True, hog_feat=True, do_threshold=True):
    
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]

    if color_space != 'RGB':
        ctrans_tosearch = convert_color(img_tosearch, color_space)
    else: 
        ctrans_tosearch = np.copy(img_tosearch)      

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64 + 8
    nblocks_per_window = (window // pix_per_cell)-1 
    nxsteps = math.ceil((nxblocks - nblocks_per_window) // cells_per_step) + 2
    nysteps = math.ceil((nyblocks - nblocks_per_window) // cells_per_step) + 2
    
    # Compute individual channel HOG features for the entire image
    if hog_channel == 0 or hog_channel == 'ALL':
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    if hog_channel == 1 or hog_channel == 'ALL':
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    if hog_channel == 2 or hog_channel == 'ALL':
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    box_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = math.ceil(yb*cells_per_step)
            xpos = math.ceil(xb*cells_per_step)

            # Extract HOG for this patch
            hog_feat1 = []
            hog_feat2 = []
            hog_feat3 = []
            if hog_channel == 0 or hog_channel == 'ALL':
               hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            if hog_channel == 1 or hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            if hog_channel == 2 or hog_channel == 'ALL':
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            if spatial_feat:
                spatial_features = bin_spatial(subimg, size=spatial_size)
            else:
                spatial_features = []

            if hist_feat:
                hist_features = color_hist(subimg, nbins=hist_bins)
            else:
                hist_features = []

            # Scale features and make a prediction
            stacked = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            
            test_features = X_scaler.transform(stacked)
            
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box = [(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)]
                box_list.append(box)



    return box_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img