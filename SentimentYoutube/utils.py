from PIL import Image
import math
from pytube import YouTube
from textblob import TextBlob
from pytube.helpers import safe_filename
import os
import shutil
import datetime
import matplotlib.pyplot as plt
from matplotlib import rc
import cv2
from IPython.display import display
from . import PythiaDemo
from . import FrameExtractor

def average(lista):
  if len(lista) == 0:
    return 0
  return sum(lista)/len(lista)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# gets the string time as HH:MM:SS and return in seconds
def string_time_int(str_time):
    segundos = int(str_time[-2:])
    segundos += 60 * int(str_time[-5:-3])
    segundos += 3600 * int(str_time[:-6])
    return segundos

def show_prediction(ulr,demo):
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

# does every preparation step before
def do_preparation(id, lang='en', frames_t = 200):

  
  ulr_to_download = 'https://www.youtube.com/watch?v=' + id
  itag = 18
  print("downloading the video from the ulr : ", ulr_to_download)
  video = YouTube(ulr_to_download)
  video.streams.get_by_itag(itag).download()

  extractor_obj = FrameExtractor(video.title,id, frames_frequency=frames_t)
  has_caption = False

  for cap in video.captions.all():

    if cap.code == lang:
      has_caption = True
      extractor_obj.has_caption = True

  extractor_obj.just_extract(video,lang)
  extractor_obj.normalize_sentiment();


  # removing the movie
  os.remove(extractor_obj.video_path)

  # enviando os dados e tornando-os visíveis no drive


  return extractor_obj