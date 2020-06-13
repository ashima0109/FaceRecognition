import os
import cv2


maindir = os.getcwd()
videodir = os.path.join(maindir, 'Data/videos')
imagepath = os.path.join(maindir, 'Data/images')

count=1


def getFrame(sec):
    """
    Extracts the frames from the video
    """
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        
        cv2.imwrite(os.path.join(imagepath, "image"+str(count)+".jpg"), image)     # save frame as JPG file
    return hasFrames


for vid in os.listdir(videodir):
    videopath = os.path.join(videodir, vid)
    vidcap = cv2.VideoCapture(videopath)

    sec = 0
    frameRate = 5 #//it will capture image in each 0.5 second

    success = getFrame(sec)
    while success:
        count = count + 1
        sec = round(sec + frameRate, 2)
        success = getFrame(sec)