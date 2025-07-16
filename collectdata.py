import cv2
import numpy as np
import time
import os

RESOLUTION = 1080,1920 # height,width

SPIRAL_K = 0.06 # keep these constant
SPIRAL_STEPS = 2500

def init_cameras():
    global cap
    for i in range(0,5):
        cap = cv2.VideoCapture(i)
        cv2.imshow(f'{i}',cap.read()[1])
        if cv2.waitKey(5000) == ord('q'):
            cv2.destroyAllWindows()
            break
        cv2.destroyAllWindows()
    global frames
    frames = []

def release_cameras():
    global cap
    cap.release()

def capture_imgs(row_pos, col_pos, timestamp = None):
    global cap
    ret, frame = cap.read()
    if ret:
        if timestamp is None:
            timestamp = time.time()
        return (frame, row_pos, col_pos, timestamp)
    else:
        raise RuntimeError("Camera capture failed")

def save_imgs(directory, frames, v=''):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for frame, row_pos, col_pos, timestamp in frames:
        filename = f"{directory}/{v}img{timestamp}_x{col_pos}_y{row_pos}.png"
        cv2.imwrite(filename, frame)
    frames = []

def calibrate(runid, frames=[], count=-1):
    screen = 255*np.ones((*RESOLUTION,3),np.uint8)
    cv2.imshow('calibrate',screen)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
        raise KeyboardInterrupt
    for row in np.linspace(0,RESOLUTION[0],11)[1:-2]:
        count += 1

        for ix, col in enumerate(np.linspace(0,RESOLUTION[1],600)[50:-50]):
            screen[:,:,:] = 255
            cv2.circle(screen,(int(col),int(row)),10,(0,0,0),-1)
            cv2.imshow('calibrate',screen)
            if ix%2 == 0:
                if ix==0:
                    if cv2.waitKey(2000) == ord('q'):
                            break
                frames += [capture_imgs(row, col)]
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            continue
        break
    cv2.destroyAllWindows()
    return frames

def validate(frames=[]):
    screen = 255*np.ones((*RESOLUTION,3),np.uint8)
    cv2.imshow('validate',screen)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
        raise KeyboardInterrupt
    for ix,rc in enumerate(np.exp((SPIRAL_K+1j)*np.linspace(-40,-7,SPIRAL_STEPS))):
        row = int((rc.real+1)*RESOLUTION[0]/2)
        col = int((rc.imag+1)*RESOLUTION[1]/2)
        screen[:,:,:] = 255
        cv2.circle(screen,(int(col),int(row)),10,(0,0,0),-1)
        cv2.imshow('validate',screen)
        if ix%4 == 0:
            if ix==0:
                if cv2.waitKey(2000) == ord('q'):
                    break
            frames += [capture_imgs(row, col)]
        
        if ix < SPIRAL_STEPS // 1.5:
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            if cv2.waitKey(3) == ord('q'):
                break
        '''
        elif ix < SPIRAL_STEPS // 4 :
            if cv2.waitKey(2) == ord('q'):
                break
        elif ix < SPIRAL_STEPS // 3:
            if cv2.waitKey(4) == ord('q'):
                break
        elif ix < SPIRAL_STEPS // 2:
            if cv2.waitKey(8) == ord('q'):
                break
        elif ix < SPIRAL_STEPS // 1.5:
            if cv2.waitKey(16) == ord('q'):
                break
        elif ix < SPIRAL_STEPS // 1.25:
            if cv2.waitKey(32) == ord('q'):
                break
        else:
            if cv2.waitKey(64) == ord('q'):
                break'''
    cv2.destroyAllWindows()
    return frames

if __name__ == "__main__":
    runid = time.time()
    init_cameras()
    frames = calibrate(runid)
    vframes = validate()
    print("Validation Process Complete. All Stimuli has been presented. Images will now be saved.")
    save_imgs(f"run_{runid}/calib", frames)
    save_imgs(f"run_{runid}/valid", vframes, v='v')
    release_cameras()