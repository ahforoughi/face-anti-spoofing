import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import cv2
#########################
from keras.models import model_from_yaml
from keras.preprocessing.image import img_to_array
#########################
from rPPG.rPPG_Extracter import *
from rPPG.rPPG_lukas_Extracter import *
import pandas as pd
import os
#########################


def get_rppg_pred(frame):
    use_classifier = True  # Toggles skin classifier
          # (Mixed_motion only) Toggles PPG detection with Lukas Kanade optical flow          
    sub_roi = []           # If instead of skin classifier, forhead estimation should be used set to [.35,.65,.05,.15]
    use_resampling = False  # Set to true with webcam 
    
    fs = 20

    timestamps = []
    time_start = [0]


    rPPG_extracter = rPPG_Extracter()

    
    dt = time.time()-time_start[0]
    time_start[0] = time.time()
    if len(timestamps) == 0:
        timestamps.append(0)
    else:
        timestamps.append(timestamps[-1] + dt)
        
    rPPG = []

    rPPG_extracter.measure_rPPG(frame,use_classifier,sub_roi) 
    rPPG = np.transpose(rPPG_extracter.rPPG)
    
        # Extract Pulse
    if rPPG.shape[1] > 10:
        if use_resampling :
            t = np.arange(0,timestamps[-1],1/fs)
            
            rPPG_resampled= np.zeros((3,t.shape[0]))
            for col in [0,1,2]:
                rPPG_resampled[col] = np.interp(t,timestamps,rPPG[col])
            rPPG = rPPG_resampled
        num_frames = rPPG.shape[1]

        t = np.arange(num_frames)/fs
    return rPPG
    

def make_pred(li):
    [single_img,rppg] = li
    single_img = cv2.resize(single_img, dim)
    single_x = img_to_array(single_img)
    single_x = np.expand_dims(single_x, axis=0)
    single_pred = model.predict([single_x,rppg])
    return single_pred

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
            images.append(filename)
    return images
    
cascPath = 'rPPG/util/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)


# load YAML and create model
yaml_file = open("trained_model/RGB_rPPG_merge_softmax_.yaml", 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights("trained_model/RGB_rPPG_merge_softmax_.h5")
print("[INFO] Model is loaded from disk")


dim = (128,128)
results_dataframe = pd.DataFrame(columns=['image_name', 'label', 'predicted_label'])
collected_results = []
counter = 0          # count collected buffers
frames_buffer = 1    # how many frames to collect to check for
accepted_falses = 1  # how many should have zeros to say it is real

# read from database
real_path = './data/real/'
fake_path = './data/fake/'
real_images = load_images_from_folder(real_path)
fake_images = load_images_from_folder(fake_path)

for image in real_images:
    print(f"[INFO] processing image: {image}")
    img = cv2.imread(real_path+image)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        # spoof detection
        for (x, y, w, h) in faces:
            sub_img = img[y:y+h,x:x+w]
            rppg_s = get_rppg_pred(sub_img)
            rppg_s = rppg_s.T

            pred = make_pred([sub_img,rppg_s])
            collected_results.append(np.argmax(pred))
            counter += 1

            if pred[0][0] > 0.5:
                predicted_label = 'real'
            else:
                predicted_label = 'fake'
            collected_results.pop(0)

        results_dataframe = results_dataframe.append({'image_name': image, 'label': 'real', 'predicted_label': predicted_label}, ignore_index=True)

for image in fake_images:
    print(f"[INFO] processing image: {image}")
    img = cv2.imread(fake_path+image)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        # spoof detection
        for (x, y, w, h) in faces:
            sub_img = img[y:y+h,x:x+w]
            rppg_s = get_rppg_pred(sub_img)
            rppg_s = rppg_s.T

            pred = make_pred([sub_img,rppg_s])
            print('========', pred[0])
            collected_results.append(np.argmax(pred))
            counter += 1
            
            if pred[0][0] > 0.5:
                predicted_label = 'real'
            else:
                predicted_label = 'fake'
            collected_results.pop(0)


        results_dataframe = results_dataframe.append({'image_name': image, 'label': 'fake', 'predicted_label': predicted_label}, ignore_index=True)


results_dataframe.to_csv('./result.csv')