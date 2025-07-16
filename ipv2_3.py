import numpy as np
import os
from PIL import Image as im
import cv2
from skimage.util import montage
import matplotlib as plt
import matplotlib.pyplot as plt



def plot(x):
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap = 'gray')
    ax.axis('off')
    fig.set_size_inches(5, 5)
    plt.show()

def montage_plot(x):
    #print(x[0].shape)
    x = np.array(x)
    #print(x.shape)
    #x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    plot(montage(x, multichannel=True))


def getImageList(directory, isValidation=False): # returns names of files (sorted by eye and webcam), number of webcams
    numWebcams = 1
    webcams = {}
    os.chdir(directory)
    for imagefile in os.listdir():
        isRightEye = "r" in imagefile
        webcamNum = "0"
        if isRightEye:
            webcamNum = imagefile.split("r")[0]
        else:
            webcamNum = imagefile.split("l")[0]
        if isValidation:
            webcamNum = webcamNum.split("V")[-1]
        result = "Right, Webcam " + webcamNum if isRightEye else \
                 "Left, Webcam " + webcamNum
        if int(webcamNum) > numWebcams:
            numWebcams = int(webcamNum)
        if result in webcams:
            old = list(webcams[result])
            webcams[result] = old + [imagefile]
        else:
            webcams[result] = [imagefile]

    return webcams, numWebcams + 1



def sort_images(imagenames):
    dWebcam = {}

    for imagename in imagenames:
        webcamNum = imagename.split("V")[-1][0]
        dWebcam[webcamNum] = dWebcam[webcamNum] + [imagename] if webcamNum in dWebcam \
                             else [imagename]
    
    return dWebcam

'''
def ip_2(images, pthr = 35, sthr = 55, s=30): # images from a SINGLE webcam

    # pthr is the threshold for detecting the pupil (low) -- FINE TUNE
    # sthr is the threshold for the sclera (higher) -- FINE TUNE

    # s = sensitivity is how much images to remove on percentile basis.
    #    s = 30 removes *about* 30% of the data set
    npupilObserved = []
    scleraObserved = []
    for imagen in images:
        image = io.imread(imagen)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # pt is the pupil thresholded image
        # st is the sclera thresholded image
        thresh, pt = cv2.threshold(gray, pthr, 1, cv2.THRESH_BINARY)
        thresh, st = cv2.threshold(gray, sthr, 1, cv2.THRESH_BINARY)

        npupilObserved += [np.sum(pt)]
        scleraObserved += [np.sum(st)]

    maxNPupil = np.percentile(npupilObserved, (100 - (s // 2))) # maximum amount of non-pupil
    minSclera = np.percentile(scleraObserved, (s * 2 // 3)) # minimum amount of sclera


    
    # now that we found the thresholds, let's implement them

    goodImages = []

    for imagen in images:    
        image = io.imread(imagen)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # pt is the pupil thresholded image
        # st is the sclera thresholded image
        thresh, pt = cv2.threshold(gray, pthr, 1, cv2.THRESH_BINARY)
        thresh, st = cv2.threshold(gray, sthr, 1, cv2.THRESH_BINARY)

        if np.sum(pt) < maxNPupil and np.sum(st) >= minSclera:
            if isValidationImage(imagen):
                imagen = imagen.split("V")[0] + imagen.split("V")[1][1:]
            else:
                imagen = imagen[1:]
            goodImages += [imagen]
    
    np.array()

    return goodImages


def saveGoodImages(imgSetN, width, height, cams, eye):
    imageSet = np.zeroes(len(imgSetN), width, height, cams)
    coords = np.zeroes(len(imgSetN))

    count = 0
    for imgN in imgSetN:
        imageNames = []
        y = ""

        if isValidationImage(imgN):
            for i in range(cams):
                comps = imgN.split(eye)
                imageNames += [f"{comps[0]}{i}{comps[1]}"]
                rest = comps[1]
                y = (int(rest.split("_")[0]),int(rest.split("_")[1].split(".")[0]))
        else:
            for i in range (cams):
                imageNames += [f"{i}{imgN[0]}"]
                y = (int(imgN[1:].split("_")[0]),int(imgN[1:].split("_")[1].split(".")[0]))

        images = [io.imread(imageName) for imageName in imageNames]

        gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

        fin = np.dstack(gray)

        imageSet[count] = fin
        coords[count] = y 

        count += 1

    np.save("AEye_data.npy", imageSet)
    np.save("AEye_labels.npy", coords)
'''
    
'''def isValidationImage(filename, hasV=True):
    if hasV:
        pieces = filename.split("V")
        if len(pieces):
            return pieces[0] != ''
    else:
        return filename.isnumeric()
        
    print("There is an issue with the nomenclature of the images.")
    return False'''


def thresholdImageSet(directory, imageList, imageType, isValidation, pthr = 35, sthr = 55, s=140): # images from a SINGLE webcam

    # pthr is the threshold for detecting the pupil (low) -- FINE TUNE
    # sthr is the threshold for the sclera (higher) -- FINE TUNE

    # s = sensitivity is how much images to remove on percentile basis.
    #    s = 30 removes *about* 30% of the data set

    # Return images that pass the threshold test.

    tempDict = {"Left, Webcam 0": (155, 202), "Left, Webcam 1": (79, 155), "Left, Webcam 2": (89, 145), \
                "Right, Webcam 0": (104, 214), "Right, Webcam 1": (76, 167), "Right, Webcam 2": (89, 145)} # These need to be fine tuned for the setting!

    pthr, sthr = tempDict[imageType]

    npupilObserved = []
    scleraObserved = []
    for imagen in imageList:
        image = np.array(im.open(f"{directory}/{imagen}"))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # pt is the pupil thresholded image
        # st is the sclera thresholded image
        thresh, pt = cv2.threshold(gray, pthr, 1, cv2.THRESH_BINARY)
        thresh, st = cv2.threshold(gray, sthr, 1, cv2.THRESH_BINARY)

        npupilObserved += [np.sum(pt)]
        scleraObserved += [np.sum(st)]

    maxNPupil = np.percentile(npupilObserved, (100 - (s // 8))) # maximum amount of non-pupil
    minSclera = np.percentile(scleraObserved, (s // 9)) # minimum amount of sclera


    # now that we found the thresholds, let's implement them

    goodImages = []

    for imagen in imageList:    
        image = np.array(im.open(f"{directory}/{imagen}"))
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # pt is the pupil thresholded image
        # st is the sclera thresholded image
        thresh, pt = cv2.threshold(gray, pthr, 1, cv2.THRESH_BINARY)
        thresh, st = cv2.threshold(gray, sthr, 1, cv2.THRESH_BINARY)

        if np.sum(pt) < maxNPupil and np.sum(st) >= minSclera:
            '''if isValidation:
                imagen = imagen.split("V")[0] + imagen.split("V")[1][1:]
            else:
                pass
                #imagen = imagen[1:]'''
            goodImages += [imagen]

    #print(goodImages)
    return goodImages


'''def listIntersection(lis):
    base = set(lis)

    for i in lis[1:]:
        base = base.intersection(i)
    
    return list(base)'''

'''
def saveGoodImages(imgSetN, width, height, cams, eye):
    imageSet = np.zeroes(len(imgSetN), width, height, cams)
    coords = np.zeroes(len(imgSetN))

    count = 0
    for imgN in imgSetN:
        imageNames = []
        y = ""

        if isValidationImage(imgN):
            for i in range(cams):
                comps = imgN.split(eye)
                imageNames += [f"{comps[0]}{i}{comps[1]}"]
                rest = comps[1]
                y = (int(rest.split("_")[0]),int(rest.split("_")[1].split(".")[0]))
        else:
            for i in range (cams):
                imageNames += [f"{i}{imgN[0]}"]
                y = (int(imgN[1:].split("_")[0]),int(imgN[1:].split("_")[1].split(".")[0]))
'''

def intersectionSets(imageList, eye, isValidation, numWebcams):
    # imageList = Dictionary e.g. {0: imageListr0, 1: imageListr1 ...} after removal of outliers
    # Find locations that exist in all of the webcams
    # Image files are named as 0l1588_542.png , for example
    # So, split by the left/right eye, take the second, and search for that location. Applies for Validation images as well.
    # Convert to grayscale, then np dstack these images
    # Return images with all three. May reconstruct based off location. No need to segregate.

    if numWebcams == 1:
        return imageList.values()

    goodImages = []
    goodLocations = []

    firstWebcam = imageList[0]

    for i in firstWebcam:
        #print(i)
        location = i.split(eye)[-1]
        locationInAllThree = True
        timestamp = ""
        if isValidation:
            timestamp = i.split("V")[0] + "V"
            #print(timestamp)

        for webcam in range(1, numWebcams):
            filename = timestamp + str(webcam) + eye + location
            #print(filename)
            if filename not in imageList[webcam]:
                locationInAllThree = False
                break

        if locationInAllThree:
            #print("Yay")
            goodImages += [np.dstack([cv2.cvtColor(np.array(im.open(timestamp + str(webcam) + eye + location)), cv2.COLOR_BGR2GRAY) for webcam in range(numWebcams)])]
            splitLocation = location[:-4].split("_")
            goodLocations += [(int(splitLocation[0]), int(splitLocation[1]))]
            #goodImages += [str(webcam) + eye + location for webcam in range(numWebcams)]
    
    return goodImages, goodLocations

    
'''
def prepareImage(imageList):
    # for each image in regular list consisting of image filenames, return the grayscale version.

    pass
'''

def main(directory, width=94, height=37, isV=False):

    imageList, numWebcams = getImageList(directory, isValidation=isV) # dictionary e.g. {"Right, Webcam 0": imageListr0 ...}
    #print(numWebcams)

    left = {} # Left eyes without outliers.
    right = {} # Right eyes without outliers

    for imageType in imageList: # imageType e.g. "Right, Webcam 0"

        thresholded = thresholdImageSet(directory, imageList[imageType], imageType, isV) # Rid of outliers.
        webcamNum = int(imageType.split(" ")[-1])

        if "Right" in imageType:
            right[webcamNum] = thresholded
        else:
            left[webcamNum] = thresholded

    #print(left)
    #print(right)
    leftData, leftLoc = intersectionSets(left, "l", isV, numWebcams)
    #print(np.array(leftData).shape)
    #print(leftLoc)
    np.save("LeftData", np.array(leftData))
    np.save("LeftLoc", np.array(leftLoc))

    rightData, rightLoc = intersectionSets(right, "r", isV, numWebcams)
    #print(rightData)
    np.save("RightData", np.array(rightData))
    np.save("RightLoc", np.array(rightLoc))

    print(str(len(leftLoc)) + " image(s) were saved.")



    '''os.chdir(directory)

    print("Sorting eyes based on webcam number...")
    dWebcam = sort_images([imgN for imgN in os.listdir() if eye in imgN])

    
    print("Calculating outliers...")
    goodImages = set([])
    for webcamNum in dWebcam.keys():
        goodImages += [ip_2(dWebcam[webcamNum])]
    
    print("Almost there...")
    imgSetN = listIntersection(goodImages)

    print(str(len(imgSetN)) + " images were found to be non-outliers. Saving...")
    saveGoodImages(imgSetN, width, height, len(dWebcam), eye)

    print("Saving complete.")'''





print("Image Processing Part 3, Version 2. Outlier Removal.")

directoryName = r"C:\Users\aksha\3W1\Run6\\"
calibDirectory = directoryName + r"calib\PostProcessing"
validDirectory = directoryName + r"valid\PostProcessing"

# eye = "l"


main(calibDirectory)
main(validDirectory, isV=True)


#np.array(im.open(f"{directory}/{imagefile}"))

'''




Process:


go through Right
go through Left



for every position, get all three webcams and store in a list W


for every list W, break W into w0, w1, w2.

compare w0, w1, and w2. must all meet in order for image to be added.



'''