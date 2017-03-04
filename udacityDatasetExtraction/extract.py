import os
import numpy as np
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
	
	lastFrame = None
	img = None
	if tuple.Label == "Car":
		frame = tuple.Frame

		# Don't read the file again if we already did
		if frame != lastFrame:
			lastFrame = frame
			img = cv2.imread("object-detection-crowdai/"+frame)

		width = tuple.xmax - tuple.xmin
		height = tuple.ymax - tuple.ymin

		# print("File: ", frame, " Size: ", (width, height))

		# Just ignore smaller images
		if width < 64 or height < 64:
			# print("Ignored")
			continue

		# Crop image from bounding box
		cropped_img = img[tuple.ymin:tuple.ymax,tuple.xmin:tuple.xmax]
		# Resize to 64x64
		resized = cv2.resize(cropped_img, (64,64))
		orig_filename, _ = os.path.splitext(frame)
		filename_prefix = str(tuple.ymin+tuple.xmin+tuple.xmin+tuple.xmax)
		filename = "out/" + filename_prefix + orig_filename + ".png"

		# If a filename already exists it's the same data labeled twice, it's alright to overwrite or skip it
		if os.path.isfile(filename):
			print("File already exists:", filename)
		else:
			cv2.imwrite(filename, resized)
			extracted += 1

		if extracted % 100 == 0:
			print("Extracted: ", extracted, " samples...")
	processed += 1
	if processed % 100 == 0:
			print("Processed: ", processed, " samples...")

print("Finished extracting")
print("Extracted: ", extracted, " samples...")
print("Processed: ", processed, " samples...")