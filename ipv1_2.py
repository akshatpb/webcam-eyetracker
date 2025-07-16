import numpy as np
import os
from PIL import Image as im
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


def getImageList(directory):
    webcams = {}
    os.chdir(directory)
    for imagefile in os.listdir():
        isRightEye = "r" in imagefile
        webcamNum = "0"
        if isRightEye:
            webcamNum = imagefile.split("r")[0]
        else:
            webcamNum = imagefile.split("l")[0]
        result = "Right, Webcam " + webcamNum if isRightEye else \
                 "Left, Webcam " + webcamNum

        if result in webcams:
            old = list(webcams[result])
            webcams[result] = old + [np.array(im.open(f"{directory}/{imagefile}"))]
        else:
            webcams[result] = [np.array(im.open(f"{directory}/{imagefile}"))]

    return webcams

def main(directory):
    instances = getImageList(directory)

    for instance in instances:
        print(instance)
        montage_plot(instances[instance])

    #print(webcams)




print("Image Processing Part 2, Version 1. Verification.")

print("Following this message, you will be prompted with an aggregate image of eyes for each of the webcams.")



directoryName = r"C:\Users\aksha\3W1\Run6\\"





calibDirectory = directoryName + r"calib\PostProcessing"
validDirectory = directoryName + r"valid\PostProcessing"




main(calibDirectory)