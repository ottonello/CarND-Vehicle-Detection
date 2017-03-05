import glob
import numpy as np
import uuid
import cv2
import matplotlib
import matplotlib.patches as patches
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

RectSize=128
ResizeTo=64

for imgFile in glob.glob("object-detection-crowdai/*.jpg"):
	img = plt.imread(imgFile)
	# im = plt.imshow(np.random.rand(10,10)*255, interpolation='nearest')
	plt.imshow(img)
	fig = plt.gcf()
	ax = plt.gca()


	# http://matplotlib.org/users/event_handling.html
	class EventHandler:
	    def __init__(self):
	        fig.canvas.mpl_connect('button_press_event', self.onPress)
	        fig.canvas.mpl_connect('motion_notify_event', self.onMove)
	        self.prevRect = None

	    def onPress(self, event):
	        if event.inaxes!=ax:
	            return

	        xStart = int(event.xdata)
	        xEnd = int(event.xdata + RectSize)
	        yStart = int(event.ydata)
	        yEnd = int(event.ydata + RectSize)
	        if yEnd <= img.shape[0] and xEnd <= img.shape[1] :
		        patch = img[yStart:yEnd,xStart:xEnd]
		        resized = cv2.resize(patch, (ResizeTo, ResizeTo))

		        filename = str(uuid.uuid1())+'.png'
		        print("Saving patch as: ", filename)
		        plt.imsave("out_nonvehicle/"+filename, resized)

	    def onMove(self, event):
	        if event.inaxes != ax: 
	        	return
	        if self.prevRect:
	        	self.prevRect.remove()

	        rect = patches.Rectangle((event.xdata,event.ydata),RectSize,RectSize,linewidth=1,edgecolor='r',facecolor='none')
	        ax.add_patch(rect)
	        fig.canvas.draw()
	        self.prevRect = rect

	handler = EventHandler()

	plt.show()