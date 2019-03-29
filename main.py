import cv2
import os
from imutils import paths
import time
import face_recognition
import pickle
from mss import mss
from PIL import Image
import numpy as np
import pandas as pd

##Importing the custom utils file
#import Project.utils


## Basic Config Settings
## The current dir of the script being run
currentdir = os.path.dirname(os.path.abspath(__file__))

## Where the face images get saved to from the screen cap
img_dir = 'ProjectImg_Larger3'

## Time in seconds to wait between screen caps
wait = 5

## Haar Cascade Face file
face_file = 'data/haarcascade_frontalface_default.xml'

## Haar Cascade Eye file
eye_file = 'C:\git\IST718\Project\data\haarcascade_eye.xml'

## Length of random characters to generate so image files dont end up having the same names
randlength = 5

## Date of the recording to append to image names
classdate = '7_18_2018'

## Number of loop iterations to go through
max_iterations = 10

## Which monitor number to use.
## For most people this will be 1.
## Use 0 to capture all screens
monnum = 2

## Path to the training data stored with each person in their own folder
### Example: FaceData/Carlo_Mencarelli/image.png
imgpath = 'FaceData/'

## Where/what to save the image encoding as
wrencode = 'FaceEncoding/IST718.pickle'

## Method to use
### cnn is: convolutional neural network - Accurate, but slow if not using CUDA
### hog is: histogram of oriented gradients - Fast, but less accurate
model = 'cnn'
#model = 'hog'

## Check for the output directory
try:
    os.makedirs(img_dir)
    print('Directory created in {}'.format(currentdir))
except OSError:
    print('Already exists in {}'.format(currentdir))


## Import the cascade files for face detection
try:
    face_cascade = cv2.CascadeClassifier(face_file)
    print('Importing cascade file {}.'.format(face_file))
except FileNotFoundError:
    print('{} doesn\'t exist.'.format(face_file))


## Import the cascade files for eye detection
try:
    eye_cascade = cv2.CascadeClassifier(eye_file)
    print('Importing cascade file {}.'.format(eye_file))
except FileNotFoundError:
    print('{} doesn\'t exist.'.format(eye_file))

## Captures the current screen and returns the image ready to be saved
## Optional parameter to set incase there's more than 1 monitor.
## Returns a raw image of the screen
def screen_cap(mnum=1):
    with mss() as sct:
        monitor = sct.monitors[mnum]
        sct_img = sct.grab(monitor)
        return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')

## Identifies faces and saves them into the imgdir directory
## Creates a temp dataframe with the Date, ElapsedSeconds, Name, and EngagementLevel
## imgfile: Image file with the faces that you want recognized.
## classdata: Date of the class
## secselapased: Number of seconds elapsed in the recording so far
## imgdir: Directory to save the individual images in
## picklefile: opened face recognition file
## Returns the temp dataframe
def cycle(imgfile, classdate, secselapsed, imgdir, picklefile, emotionpickle):
    tempemotionframe = pd.DataFrame(columns=['Date', 'ElapsedSeconds', 'Name', 'EngagementLevel'])
    img = cv2.imread(imgfile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=10, minSize=(50,50), flags=cv2.CASCADE_SCALE_IMAGE)
    #randletters = Project.utils.rand_string(randlength)
    for (x,y,w,h) in faces:
        sub_face = img[y:y + h+2, x:x + w+2]
        name = recognize(sub_face, picklefile, modl=model)
        if name is not "Unknown":
            #print(name)
            emotion = emotionrec(sub_face, emotionpickle, modl=model)
            eyes = len(eye_cascade.detectMultiScale(sub_face))
            FaceFileName = imgdir + "/" + name + '_' + str(classdate) + "_" + str(secselapsed) + "_" + str(emotion) + "_" + str(eyes) + ".jpg"
            cv2.imwrite(FaceFileName, sub_face)
            tempemotionframe.loc[len(tempemotionframe)] = [classdate, secselapsed, name, emotion]
        else:
            pass
            #print("Skipping Unknown")
    return tempemotionframe

## Recognizes the person in the image
## rawimg: The raw image data that's captured after recognizing a face using CV
## pickledata: opened face recognition file
## modl: Which model to use. CNN or HOG. Defaults to CNN right now
def recognize(rawimg, emotionpickle, modl='cnn'):
    boxes = face_recognition.face_locations(rawimg, model=modl)
    encodings = face_recognition.face_encodings(rawimg, boxes)

    names = []

    for encode in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(emotionpickle["encodings"], encode)
        name = "Unknown"

        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = emotionpickle["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)
    if len(names) > 0:
        name = names[0].replace(' ', '_').lower()
    else:
        name = 'Unknown'
    return name

## Temp function that returns the emotion of the raw img that's fed into it
## Currently returns a random int
def emotionrec(rawimg, pickledata, modl='cnn'):
    boxes = face_recognition.face_locations(rawimg, model=modl)
    encodings = face_recognition.face_encodings(rawimg, boxes)

    scores = []

    for encode in encodings:
        matches = face_recognition.compare_faces(pickledata["encodings"], encode)
        score = 0
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            for i in matchedIdxs:
                name = pickledata["names"][i]
                if name == 'positive':
                    score = 1.5
                elif name == 'negative':
                    score = -1
                else:
                    score = 0
                scores.append(score)
    if len(scores) > 0:
        return sum(scores)
    else:
        return 0

## Setting the iterator to 0
i = 0

## Loading the facial recognition encoding
data = pickle.loads(open('FaceEncoding/IST718.pickle', 'rb').read())
emotiondata = pickle.loads(open('C:\\git\\IST718\\Project\\emotion_rec\\simple_emotion.pickle', 'rb').read())

## Initializing the empty dataframe
emotionframe = pd.DataFrame(columns=['Date', 'ElapsedSeconds', 'Name', 'EngagementLevel'])

while i < max_iterations:
    name = 'temp.jpg'
    tempimg = screen_cap(mnum=monnum).save(name)
    elapsed = i * wait
    count = cycle(imgfile=name, classdate=classdate, secselapsed=elapsed, imgdir=img_dir, picklefile=data, emotionpickle=emotiondata)
    emotionframe = pd.concat([emotionframe, count], ignore_index=True)
## Using the CNN model with CPU, there's more procesing time which means the capture eventually beings to lag behind
## If using GPU and CNN model then there's no need to wait extra or less time
    if model is 'cnn':
        time.sleep(wait)
    else:
        time.sleep(wait)

    i += 1

print(emotionframe.head())