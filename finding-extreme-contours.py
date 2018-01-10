
# coding: utf-8

# # Finding Extreme Contours

# We will use [Convex Hull](https://en.wikipedia.org/wiki/Convex_hull) method to to find the extreme points in the image. 
# 
# We will segment the skin/hand from the image and then find the extreme points along the convex hull.
# 
# Let's Begin.

# In[14]:

import imutils
import cv2


# We will now load the image and covert it to grayscale and then blur it slightly.

# In[27]:

image = cv2.imread("hand.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)


# ## Morphological Operations

# Now we we will threashold the image and perforn erosions and dilations to remove regions of noise
# 
# [Erosion and Dialation](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)

# In[28]:

thresh = cv2.threshold(gray,45,255,cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh,None,iterations=2)
thresh = cv2.dilate(thresh,None,iterations=2)


# ## Finding Contours
# 
# Contour is a Numpy array of x,y co-ordinates.

# In[29]:

cnts = cv2.findContours(thresh.copy(),
                        cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
c = max(cnts, key=cv2.contourArea)


# We will now find the extreme left, right, top and bottom coordinates.

# In[30]:

extLeft = tuple(c[c[:,:,0].argmin()][0])
extRight = tuple(c[c[:,:,0].argmax()][0])
extTop = tuple(c[c[:,:,1].argmin()][0])
extBot = tuple(c[c[:,:,1].argmax()][0])


# ## Outline the Hand

# In[31]:

# draws outline of hand
cv2.drawContours(image, [c], -1, (0,255,255), 2)
# draws circle on extreme points
cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
cv2.circle(image, extRight, 8, (0, 255, 0), -1)
cv2.circle(image, extTop, 8, (255, 0, 0), -1)
cv2.circle(image, extBot, 8, (255, 255, 0), -1)


# ## Display Image

# In[32]:

cv2.imshow("Image", image)
cv2.waitKey(0)

