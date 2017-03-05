import glob
import sys
import numpy as np
import uuid
import cv2
import pandas
import matplotlib
import matplotlib.patches as patches
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# Extract same N as vehicles
ExtractN=32200

# Start of the sliding window search
YStart = 464
YStop = 656

# Rect sizes
RectSize=128
ResizeTo=64

# Steps for sliding window search
XStep = RectSize
YStep = RectSize

csv = pandas.read_csv("object-detection-crowdai/labels.csv")
print("Total: ", len(csv), " samples")

# Continue to next frame

# Bonus: find only below half the screen
# Bonus: check files that are not in the list(no cars)

extracted = 0
curFrame = None
curKnownCarRects = []

for tuple in csv.itertuples():
	if extracted > ExtractN:
		print("Done, extracted: ", extracted)
		sys.exit(0)

	if curFrame != None and curFrame != tuple.Frame:
		# When the frame changes, find places with RectSize x RectRize that aren't within the known rects,
		# Read the image file and save those resized rects -> sliding window?
		img = plt.imread("object-detection-crowdai/"+curFrame)
		# Do sliding window search from YStart to YStop, from 0 to image width
		xStart = 0
		xEnd = img.shape[1]
		for y1 in range(YStart, YStop, YStep):
			for x1 in range(xStart, xEnd, XStep):
				valid = True
				x2 = x1 + RectSize
				y2 = y1 + RectSize
				for rect in curKnownCarRects:
					noOverlap = (x1 > rect[2] or x2 < rect[0]) or (y1 > rect[3] or y2 < rect[1])
					if not noOverlap:
						valid = False
						break

				if valid:
					if y2 <= img.shape[0] and x2 <= img.shape[1]:
						patch = img[y1:y2,x1:x2]
						resized = cv2.resize(patch, (ResizeTo, ResizeTo))

						filename = str(uuid.uuid1())+'.png'
						# print("Saving patch as: ", filename)
						extracted += 1
						plt.imsave("out_nonvehicle_2/"+filename, resized)
						if extracted % 100 == 0:
							print("Extracted ", extracted, " of ", ExtractN)

		# Update current frame and rects
		curFrame = tuple.Frame
		curKnownCarRects = []
	# Read entries and save the rect
	curKnownCarRects.append([tuple.xmin, tuple.ymin, tuple.xmax, tuple.ymax])
	curFrame = tuple.Frame