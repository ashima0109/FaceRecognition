import os
import cv2


maindir = os.getcwd()
imagedir = os.path.join(maindir, 'Data/images')

imagepath = os.path.join(imagedir, 'image150.jpg')
# load the photograph
pixels = cv2.imread(imagepath)

# load the pre-trained model
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# perform face detection
bboxes = classifier.detectMultiScale(pixels, 1.1, 5)
# print bounding box for each detected face
for box in bboxes:
	# extract
	x, y, width, height = box
	x2, y2 = x + width, y + height
	# draw a rectangle over the pixels
	cv2.rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
# # show the image
# cv2.imshow('face detection', pixels)
# # keep the window open until we press a key
# cv2.waitKey(0)
# # close the window
# cv2.destroyAllWindows()

cv2.imwrite('faces.jpg', pixels)