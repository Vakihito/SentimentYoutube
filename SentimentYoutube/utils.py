from PIL import Image
import math
import os
import shutil
import datetime
import matplotlib.pyplot as plt
from matplotlib import rc
import cv2
from IPython.display import display
import ktrain
from .PythiaDemo import PythiaDemo

print("Loading pythia, this may take a while ...")
demo = PythiaDemo()

print("Loading predictors for face analasys ...")
predictor_face = ktrain.load_predictor('/content/sentiment_face')

print("Loading predictors for text analasys ...")
predictor_text = ktrain.load_predictor('/content/sentiment_text')

def average(lista):
  if len(lista) == 0:
    return 0
  return sum(lista)/len(lista)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Predicts a text (positive or negative )  
def predict_text(description, return_proba=True):
    return predictor_text.predict(description, return_proba=True)

# Predicts a face in a image (positive or negative )
def predict_face(description, return_proba=True):
    return predictor_face.predict(description, return_proba=True)


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