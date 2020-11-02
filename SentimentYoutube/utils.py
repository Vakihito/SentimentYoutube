from PIL import Image
import requests
from io import BytesIO
import math
import os
import shutil
import datetime
import matplotlib.pyplot as plt
from matplotlib import rc
import cv2
from IPython.display import display
import ktrain
import numpy as np
from .PythiaDemo import PythiaDemo
from google.colab.patches import cv2_imshow

model_load = False


def load_models(face_dir='/content/sentiment_face',text_dir='/content/sentiment_text', net_model_dir="/content/res10_300x300_ssd_iter_140000.caffemodel",net_prototx_dir="/content/deploy.prototxt.txt"):
  global model_load
  if (not model_load):
    print("Loading pythia, this may take a while ...\n\n")
    global demo
    demo = PythiaDemo()

    print("Loading predictors for face analasys ...\n")
    global predictor_face
    predictor_face = ktrain.load_predictor(face_dir)

    print("Loading predictors for text analasys ...\n")
    global predictor_text
    predictor_text = ktrain.load_predictor(text_dir)
    model_load = True

    # load our serialized model from disk
    print("Loading model net model...")
    global net
    net = cv2.dnn.readNetFromCaffe(net_prototx_dir, net_model_dir)

def prop_of_having_face(path,confidence_arg=0.8,show_img=False):
    image = cv2.imread(path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))


    # pass the blob through the network and obtain the detections and
    # predictions
    # print("[INFO] computing object detections...")

    net.setInput(blob)
    detections = net.forward()


    certeza = 0

    # loop over the detections
    for i in range(0, detections.shape[1]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > confidence_arg:
            certeza = confidence
            if(not show_img):
                return confidence
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
    
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # show the output image


    if (show_img):
        cv2_imshow( image)
        cv2.waitKey(0)

    return certeza

def explain_frame(path):
    plt.imshow( predictor_face.explain(path) )
    plt.show()
    frame_des = show_prediction(path)
    return predictor_text.explain(frame_des)
def explain_text(text):
    return predictor_text.explain(text)

def average(lista):
  if len(lista) == 0:
    return 0
  return sum(lista)/len(lista)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Predicts a text (positive or negative )  
def predict_text(description, return_prob=True):
    return predictor_text.predict(description, return_proba=return_prob)

# Predicts a face in a image (positive or negative )
def predict_face(path, return_prob=True):
    prob = predictor_face.predict_filename(path, return_proba=return_prob)[0]
    size_prob = len(prob)
    maxi = 0
    max_idx = -1
    for i in range(size_prob):
        if maxi < prob[i]:
            maxi = prob[i]
            max_idx = i

    # 0 - negativo
    # 1 - neutro
    # 2 - positivo
    if max_idx == 0:
        return -maxi
    if max_idx == 2:
        return maxi
    return 0


# gets the string time as HH:MM:SS and return in seconds
def string_time_int(str_time):
    segundos = int(str_time[-2:])
    segundos += 60 * int(str_time[-5:-3])
    segundos += 3600 * int(str_time[:-6])
    return segundos

def show_prediction(ulr):
    tokens = demo.predict(ulr)
    answer = demo.caption_processor(tokens.tolist()[0])["caption"]
    return answer

def show_image_url(ulr):
    response = requests.get(ulr)
    img = Image.open(BytesIO(response.content))
    img.thumbnail((256,256), Image.ANTIALIAS)
    display(img)
    return img

def show_image(path):
  img = Image.open(path)
  img.thumbnail((256,256), Image.ANTIALIAS)
  display(img)
  return img

# receives the seconds as a int and return a string in HH:MM:SS
def seconds_to_time(sec):
  seconds = int(sec)
  min = int(seconds / 60)
  hours = int(min / 60)
  min = min % 60
  
  hours = str(hours)
  min = str(min)
  seconds = str(sec % 60)
  if len(hours) < 2 :
    hours = '0' + hours
  if len(min) < 2 :
    min = '0' + min
  if len(seconds) < 2:
    sec = '0' + seconds
  return hours + ':' + min + ':' + seconds