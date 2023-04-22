import opendatasets as od
import pandas
import cv2
import os
import time
import sys
import numpy as np
import re
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

def main(video_path):
    # Do something with the video file
    print(f"Processing video at {video_path}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file.")
    parser.add_argument("video_path", type=str, help="path to the video file")
    args = parser.parse_args()

    main(args.video_path)
count =0
frame_count=0
# Create the folder if it doesn't exist
if not os.path.exists("capimages/satellite"):
    os.makedirs("capimages/satellite")

video = cv2.VideoCapture("output_video.mp4")
success, image = video.read()
# Loop through the video frames
while success:
    cv2.imshow('Video', image)
    if count % 4 == 0 and frame_count < 3:
        filename = f'capimages/satellite/frame{count}_sat.jpg'
        cv2.imwrite(filename, image)
        print(f'Frame {count} saved')
        frame_count += 1

    
    elif count % 4 != 0:
        if count % 30 == 0:
            filename = f'capimages/satellite/frame{count}_sat.jpg'
            cv2.imwrite(filename, image)
            print(f'Frame {count} saved')
    else:
        frame_count = 0

    # Read the next frame
    success, image = video.read()
    count += 1

    # Wait for a millisecond between frames
    time.sleep(1/30)
    if cv2.waitKey(1) == ord('q'):
        break
# Release the video file
video.release()
cv2.destroyAllWindows()

# predict images
framObjTest = {'img' : []
          }

def LoadData( frameObj = None, imgPath = None, maskPath = None, shape = 128):
    imgNames = os.listdir(imgPath)
    maskNames = []
    
    ## generating mask names
    for mem in imgNames:
        mem = mem.split('_')[0]
        if mem not in maskNames:
            maskNames.append(mem)
    
    imgAddr = imgPath + '/'
    maskAddr = maskPath + '/'
    
    for i in range (len(imgNames)):
        try:
            img = plt.imread(imgAddr + maskNames[i] + '_sat.jpg') 
            mask = plt.imread(maskAddr + maskNames[i] + '_mask.png')
            
        except:
            continue
        img = cv2.resize(img, (shape, shape))
        mask = cv2.resize(mask, (shape, shape))
        frameObj['img'].append(img)
        frameObj['mask'].append(mask[:,:,0]) # this is because its a binary mask and img is present in channel 0
        
    return frameObj

def LoadDatatest( frameObj = None, imgPath = None, shape = 128):
    imgNames = os.listdir(imgPath)
    maskNames = []
    
    ## generating mask names
    for mem in imgNames:
        mem = mem.split('_')[0]
        if mem not in maskNames:
            maskNames.append(mem)
    
    imgAddr = imgPath + '/'
    
    for i in range (len(imgNames)):
        try:
            img = plt.imread(imgAddr + maskNames[i] + '_sat.jpg') 
            
        except:
            continue
        img = cv2.resize(img, (shape, shape))
        frameObj['img'].append(img)
        
    return frameObj

model = tf.keras.models.load_model('MapSegmentationGenerator.h5')

framObjTest = LoadDatatest( framObjTest, imgPath = 'capimages/satellite'
                         , shape = 128)
def predict17 (valMap, model, shape = 128):
    img = valMap['img']
    imgProc = img
    imgProc = np.array(img)
    
    predictions = model.predict(imgProc)
  

    return predictions, imgProc


def Plotter17(predMask):
    plt.imshow(predMask)


save_dir = '/home/worakan/save_images_cansat'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

sixteenPrediction, actuals = predict17(framObjTest, model)
# save the predicted mask images
for i, predMask in enumerate(sixteenPrediction):
    img = actuals[i]
    Plotter17(predMask)
    plt.savefig(os.path.join(save_dir, f'prediction_{i}.jpg'))
    plt.close()

output_name = "predictoutput_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

image_folder = "/home/worakan/save_images_cansat"
image_names = sorted(os.listdir(image_folder))

image_path = os.path.join(image_folder, image_names[0])
image = cv2.imread(image_path)
height, width, channels = image.shape

video = cv2.VideoWriter(output_name, fourcc, 30, (width, height))

for image_name in image_names:
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    video.write(image)

video.release()
cv2.destroyAllWindows()
os.system(f"ffplay -autoexit {output_name}")
