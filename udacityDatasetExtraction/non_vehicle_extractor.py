import numpy as np
import uuid
import matplotlib
import matplotlib.patches as patches
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

img = plt.imread("object-detection-crowdai/1479498371963069978.jpg")
# im = plt.imshow(np.random.rand(10,10)*255, interpolation='nearest')
plt.imshow(img)
fig = plt.gcf()
ax = plt.gca()

RectSize=256
ResizeTo=64

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
        patch = img[yStart:yEnd,xStart:xEnd]
        filename = str(uuid.uuid1())+'.png'
        print("Saving patch as: ", filename)
        plt.imsave("out_nonvehicle/"+filename, patch)

    def onMove(self, event):
        if event.inaxes != ax: 
        	return
        if self.prevRect:
        	self.prevRect.remove()

        rect = patches.Rectangle((event.xdata,event.ydata),256,256,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        fig.canvas.draw()
        self.prevRect = rect

handler = EventHandler()

plt.show()