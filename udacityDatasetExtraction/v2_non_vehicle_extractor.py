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

# Number of samples to extract in total
# Extract same N as vehicles
ExtractN=29190
# Max umber of samples to extract per file
SamplesPerFile=20

# Start of the sliding window search
YStart = 600
YStop = 996

# Rect sizes
RectSize=128
ResizeTo=64

# Steps for sliding window search
XStep = RectSize
YStep = RectSize

csv = pandas.read_csv("object-dataset/labels.csv", sep=' ') 
print("Total: ", len(csv), " samples")

# Continue to next frame

# Bonus: find only below half the screen
# Bonus: check files that are not in the list(no cars)

extracted = 0
curFrame = None
extractedThisFrame = 0
# Rects of known objects
curKnownCarRects = []

def extract(img, xStart, xEnds):
	global extractedThisFrame, extracted
	if extractedThisFrame >= SamplesPerFile:
		return
	
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
			extractedThisFrame += 1
			plt.imsave("out_nonvehicle_2/"+filename, resized)
			if extracted % 100 == 0:
				print("Extracted ", extracted, " of ", ExtractN)
	return 


for tuple in csv.itertuples():
	if extracted > ExtractN:
		print("Done, extracted: ", extracted)
		sys.exit(0)

	if curFrame != None and curFrame != tuple.Frame:
		# When the frame changes, find places with RectSize x RectRize that aren't within the known rects,
		# Read the image file and save those resized rects -> sliding window?
		img = plt.imread("object-dataset/"+curFrame)
		# Do sliding window search from YStart to YStop, from 0 to image width
		xStart = int(img.shape[1]/2)
		xEnd = img.shape[1]
		for y1 in range(YStart, YStop, YStep):
			for x1 in range(xStart, xEnd, XStep):
				extract(img, xStart, xEnd)

		# Update current frame and reset rects
		curFrame = tuple.Frame
		curKnownCarRects = []
		extractedThisFrame = 0

	# Read entries and save the rect
	curKnownCarRects.append([tuple.xmin, tuple.ymin, tuple.xmax, tuple.ymax])
	curFrame = tuple.Frame