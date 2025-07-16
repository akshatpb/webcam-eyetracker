import numpy as np
import os
from PIL import Image as im
#import face_recognition
from skimage import io
import cv2
import dlib
import time




def split_image(image, lcamLayout):
    squares = list(image.shape)
    #print(squares)
    images = []
    # lcamlayout is [1, 3]
    for i in range(lcamLayout[0]): # rows = [0]
        for j in range(lcamLayout[1]): # columns = [0,1,2]
            '''print(j + 1)
            print(lcamLayout[1])
            print(squares[1])
            print(str(j/lcamLayout[1]*squares[1]) + " to " + str((j+1)/lcamLayout[1]*squares[1]))'''
            images += [image[round(i/lcamLayout[0]*squares[0]):round((i+1)/lcamLayout[0]*squares[0]), \
                             round(j/lcamLayout[1]*squares[1]):round((j+1)/lcamLayout[1]*squares[1]),:]]
    #print(len(images))
    #print(type(images[1]))
    '''cv2.imshow("", images[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    return images



    # it should be 0 to 1500, 0 to 1221
    #              0 to 1500, 1222 to 2442
    #              0 to 1500, 2443 to 3664
    

'''def crop_for_eyes(image, width=37, height=22):
    uimage = image.astype("uint8")
    face_landmarks = face_recognition.face_landmarks(uimage)[0]
 
    eyes = []
 
    for facial_feature in list(face_landmarks.keys())[5:7]:
        yMin = min(list(map(lambda x: x[0], face_landmarks[facial_feature])))
        yMax = max(list(map(lambda x: x[0], face_landmarks[facial_feature])))
        xMin = min(list(map(lambda x: x[1], face_landmarks[facial_feature])))
        xMax = max(list(map(lambda x: x[1], face_landmarks[facial_feature])))
 
        eye = image[xMin:xMax, yMin:yMax]

        eyedata = im.fromarray(eye)
        #eyedata = eyedata.resize((width, height))
        
        eyes += [eyedata]

    return eyes[0], eyes[1] # return left, right eye'''


def crop_for_eyes(image, width=94, height=37):
    # Convert image into grayscale
    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    face = detector(gray)[0]

    landmarks = predictor(image=gray, box=face)

    xs = []
    ys = []
    for n in range(36,42): # Left Eye
        xs += [landmarks.part(n).y]
        ys += [landmarks.part(n).x]

    yMin = min(ys)
    yMax = max(ys)
    xMin = min(xs)
    xMax = max(xs)

    eye = image[xMin:xMax, yMin:yMax]
    lefteye = im.fromarray(eye)
    lefteye = lefteye.resize((width, height))

    xs = []
    ys = []
    for n in range(42,48): # Right  Eye
        xs += [landmarks.part(n).y]
        ys += [landmarks.part(n).x]

    yMin = min(ys)
    yMax = max(ys)
    xMin = min(xs)
    xMax = max(xs)


    eye = image[xMin:xMax, yMin:yMax]
    righteye = im.fromarray(eye)
    righteye = righteye.resize((width, height))

    return lefteye, righteye # return left, right eye



'''def isValidationImage(pieces):
    if len(pieces):
        return pieces[0] != ''
    print("There is an issue with the nomenclature of the images.")
    return False'''

def ip(Limage, filename, path, lcamLayout, isValidation=False):
    images = split_image(Limage, lcamLayout)

    for camNum, image in enumerate(images):
        try:
            left, right = crop_for_eyes(image) # may adjust for width, height
        except:
            return True

        timestamp, x, y = filename.split("_")
        timestamp = timestamp[4:] if isValidation else None
        x = round(float(x[1:]))
        y = round(float(y[1:-4]))

        if isValidation:
            left.save(f"{path}{timestamp}V{camNum}l{x}_{y}.png")
            right.save(f"{path}{timestamp}V{camNum}r{x}_{y}.png")
        else:
            left.save(f"{path}{camNum}l{x}_{y}.png")
            right.save(f"{path}{camNum}r{x}_{y}.png")
    return False

        # Old image filename format:
        #
        # img1671933751.47774_x336.56093489148583_y108.0
        # vimg1671933841.34595_x895_y507
        #
        #
        # Image filename format:
        # 
        # 0l1588_542.png
        # 1l1588_542.png
        # 2l1588_542.png
        # 0r1588_542.png
        # 1r1588_542.png
        # 2r1588_542.png
        #
        # 134V0l1588_542.png
        # 134V1l1588_542.png
        # 134V2l1588_542.png
        # 134V0r1588_542.png
        # 134V1r1588_542.png
        # 134V2r1588_542.png





def main(foldername, lcamLayout, isV=False):
    # images = {}
    #print(os.path)
    #print(os.listdir())
    #print(os.getcwd())
    try:
        os.chdir(r"C:\Users\aksha\3W1\\" + foldername)
        images = os.listdir()
        numImages = len(images)
        os.mkdir("PostProcessing")
    except:
        print("Error: Invalid directory / folder name.")
        return
    #os.chdir(r"C:\Users\aksha\3W1\\" + foldername)
    path = os.getcwd()
    print("Progress:")
    #print(path)
    numFailure = 0
    start_time = time.time()
    for num, imagefile in enumerate(images):
        print(str(round(num / numImages * 100, 1)) + "%")
        if num % 100 == 0 and num != 0:
            remaining_time = (numImages - num) * ((time.time() - start_time) / num)
            print(f"Approximately " + str(round(remaining_time / 60)) + " minutes remaining. (ETA: " + str(time.strftime("%H:%M", time.localtime(time.time() + remaining_time))) + ")")
        failure = ip(np.array(im.open(f"{path}/{imagefile}")), imagefile, path + r"\PostProcessing\\", lcamLayout, isValidation=isV)
        if failure:
            numFailure += 1
            print(f"{numFailure} image(s) detected without eyes")
    return numImages, numFailure


print("Image Processing Part 1, Version 2. Eye Cropping.")

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

foldername = "Run6"
numWebcams = 3
lcamlayout = [1, 1] # row, column
if numWebcams == 3:
    lcamlayout = [1, 3]  # row, column
if numWebcams == 4:
    lcamlayout = [2, 2] # row, column
 
print("Calibration (Training) Images: ")
numImagesC, numFailureC = main(foldername + r"\calib", lcamlayout)
print("Stimulus (Validation) Images: ")
numImagesV, numFailureV = main(foldername + r"\valid", lcamlayout, isV=True)

print("Caibration: " + str(numImagesC - numFailureC) + " image(s) processed, with " + str(numFailureC) + " failure(s).")
print("Validation: " + str(numImagesV - numFailureV) + " image(s) processed, with " + str(numFailureV) + " failure(s).")

'''
Final: 94 x 37

133 x 54
143 x 57
143 x 55
138 x 60

170 x 64
169 x 62
166 x 60
167 x 65


131 x 54
140 x 61
132 x 56
135 x 55


97 x 48
102 x 62
103 x 50
96 x 54
104 x 63

118 x 44
112 x 44
109 x 43
111 x 42
115 x 46


116 x 39
124 x 39
121 x 40
136 x 49
132 x 47'''