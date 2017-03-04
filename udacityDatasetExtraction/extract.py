import os
#import matplotlib.image as mpimg
import cv2
import pandas

"""
Extracts car samples from Udacity dataset from:
https://github.com/udacity/self-driving-car/tree/master/annotations

Note: Dataset one has at this point wrong labels, they are:
xmin,xmax,ymin,ymax
but should be:
xmin,ymin,xmax,ymax
"""
csv = pandas.read_csv("object-detection-crowdai/labels.csv")
print("Total: ", len(csv), " samples")

extracted = 0
processed = 0
for tuple in csv.itertuples():
	
	if tuple.Label == "Car":
		frame = tuple.Frame
		img = cv2.imread("object-detection-crowdai/"+frame)
		# Crop image from bounding box
		cropped_img = img[tuple.ymin:tuple.ymax,tuple.xmin:tuple.xmax]
		orig_filename, _ = os.path.splitext(frame)
		filename_prefix = str(tuple.ymin+tuple.xmin+tuple.xmin+tuple.xmax)
		filename = "out/" + filename_prefix + orig_filename + ".png"
		if os.path.isfile(filename):
			print("File already exists:", filename)
		cv2.imwrite(filename, cropped_img)
		extracted += 1
		if extracted % 100 == 0:
			print("Extracted: ", extracted, " samples...")
	processed += 1
	if processed % 100 == 0:
			print("Processed: ", processed, " samples...")

print("Finished extracting")
print("Extracted: ", extracted, " samples...")
print("Processed: ", processed, " samples...")