from pytube import YouTube
from textblob import TextBlob
from pytube.helpers import safe_filename
import numpy as np
import math
import os
import shutil
import math
import datetime
import matplotlib.pyplot as plt
from matplotlib import rc
import cv2
from .frame_caption import frame_caption
from .frame_struct import frame_struct
from .caption_struct import caption_struct
from .utils import *


class FrameExtractor():
    '''
    Class used for extracting frames from a video file.
    '''
    def __init__(self, video_name, video_id, video_path="/content/vqa-maskrcnn-benchmark/", video_dir="./dataset/", frames_frequency=200):
        self.video_name = safe_filename(video_name)
        self.video_id = video_id
        self.video_path = video_path + self.video_name + ".mp4"
        self.video_dir = video_dir + self.video_name  + self.video_id + "_dir"
        self.img_name = self.video_name + "_img"

        self.frames_frequency = frames_frequency
        self.vid_cap = cv2.VideoCapture(self.video_path)
        self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

        self.key_time = []  # saves the time of captions
        self.captions_save = [] # this will hold the captions in a viable format
        self.caption_str = "" #this will save the caption_str
        self.time_frames = {} # correlates a time with a frame
        self.frames_captions = [] # holds the corrlation between frame and caption
        self.has_caption = False

        
    def get_video_duration(self):
        duration = self.n_frames/self.fps
        print(f'Duration: {datetime.timedelta(seconds=duration)}')


    def format_caption(self, caption_string):
        '''
          extract the information from the caption and saves in a caption struct 
        '''
        line_counter = 1
        for line in caption_string.split('\n'):
            # time type
            if (line_counter + 2) % 4  == 0:
                line_aux = line.split(" --> ")
                time_s = string_time_int(line_aux[0][:-4])
                time_e = string_time_int(line_aux[1][:-4])
            # comment type
            if (line_counter + 1) % 4  == 0:  
                string_cap = line
                self.captions_save.append(caption_struct(time_s, time_e, string_cap))
                self.key_time.append(time_s)
                self.key_time.append(time_e)
            line_counter += 1

        self.key_time = sorted(set(self.key_time))
    

    def extract_frames_no_caption(self, img_ext = '.jpg'):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
        
        if self.video_dir is None:
            self.video_dir = os.getcwd()
        else:
            if not os.path.isdir(self.video_dir):
                os.mkdir(self.video_dir)
                print(f'Created the following directory: {self.video_dir}')
                
        
        frame_cnt = 0
        img_cnt = 0

        while self.vid_cap.isOpened():
            
            success,image = self.vid_cap.read() 
            
            if not success:
                break
            
            if frame_cnt % self.frames_frequency == 0:
                img_path = os.path.join(self.video_dir, ''.join([self.img_name, '_', str(img_cnt), img_ext]))
                
                print("image path: ", img_path)
                cv2.imwrite(img_path, image) 

                self.time_frames[int((frame_cnt / self.fps))] = frame_struct(int((frame_cnt / self.fps)),img_path, show_prediction(img_path))  
                img_cnt += 1
                
            frame_cnt += 1
        
        self.vid_cap.release()
        cv2.destroyAllWindows()
    def extract_frames_caption(self, img_ext='.jpg'):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
        
        if self.video_dir is None:
            self.video_dir = os.getcwd()
        else:
            if not os.path.isdir(self.video_dir):
                os.mkdir(self.video_dir)
                print(f'Created the following directory: {self.video_dir}')
        
        frame_cnt = 0
        img_cnt = 0
        caption_counter = 0
        last_time = -1

        while self.vid_cap.isOpened():
            
            success,image = self.vid_cap.read() 
            
            if not success:
                break

            # making a frame for each time of caption
            if ( len(self.key_time) > caption_counter and int((frame_cnt / self.fps)) ==  self.key_time[caption_counter]) and (last_time != self.key_time[caption_counter]):
                last_time = self.key_time[caption_counter]
                img_path = os.path.join(self.video_dir, ''.join([self.img_name, '_', str(img_cnt), img_ext]))
                
                print("image path: ", img_path)
                cv2.imwrite(img_path, image)

                self.time_frames[last_time] = frame_struct(last_time,img_path, show_prediction(img_path))

                

                img_cnt += 1
                caption_counter += 1

            elif frame_cnt % self.frames_frequency == 0:
                img_path = os.path.join(self.video_dir, ''.join([self.img_name, '_', str(img_cnt), img_ext]))
                
                print("image path: ", img_path)
                cv2.imwrite(img_path, image)  

                self.time_frames[int((frame_cnt / self.fps))] = frame_struct(int((frame_cnt / self.fps)),img_path, show_prediction(img_path))
                

                img_cnt += 1

            frame_cnt += 1
        
        self.vid_cap.release()
        cv2.destroyAllWindows()

    def correlates_caption_frame(self):
      for cap in self.captions_save:
        self.frames_captions.append(frame_caption(
                                    self.time_frames[cap.start],
                                    self.time_frames[cap.end],
                                    cap))
        # gets the time of each frame generated
      list_frames_times = list(self.time_frames.keys())
      
      time_frames_without_caption = []
      # remove from this list the frames are already in the frames_captions
      for time_of_each_frame in list_frames_times:
        # check if the time of the frame is in the captions time
        if time_of_each_frame not in self.key_time:
          time_frames_without_caption.append(time_of_each_frame)
      
      # insert the frames without captions
      # run through the frame_caption list
      pointer = 0
      insert_frames = 0
      while(pointer < len(self.frames_captions) and insert_frames < len(time_frames_without_caption)):
        if (self.frames_captions[pointer].caption.start > time_frames_without_caption[insert_frames]):
          aux_frame_caption = frame_caption(self.time_frames[time_frames_without_caption[insert_frames]])
          self.frames_captions.insert(pointer, aux_frame_caption)
          
          insert_frames +=1

        pointer +=1
      
      # insert in the end all frames that where not added
      for frames_to_add  in range(insert_frames, len(time_frames_without_caption)):
        aux_frame_caption = frame_caption(self.time_frames[time_frames_without_caption[frames_to_add]])
        self.frames_captions.append(aux_frame_caption)

    def just_extract(self, vid_cap,lang="en"):
      
      captions=None
      if (self.has_caption):
        # 94 - blue
        print("\033[92mExtracting with captions\033[0m")
        captions  = vid_cap.captions.get_by_language_code(lang)
        captions_str = str(captions.generate_srt_captions())
        self.caption_str = captions_str
        self.format_caption(captions_str)
        self.extract_frames_caption()
        self.correlates_caption_frame()
      else :
        # 93 - yellow
        print("\033[91mWarning : extracting without captions\033[0m")
        self.extract_frames_no_caption()

    def show_descriptions_from_all_frames(self, show_img=True):
        for time_of_frame in self.time_frames:
          self.time_frames[time_of_frame].printF(show_img)
          print("\n############################################\n")

    def normalize_sentiment(self):
        if self.has_caption:
            max_value_caption_ktrain = max([abs(cap.feeling) for cap in self.captions_save])
            for i in range(len(self.captions_save)):
                self.captions_save[i].feeling = self.captions_save[i].feeling / max_value_caption_ktrain
        
        max_value_frame_ktrain = max([abs(self.time_frames[min_f].feeling_ktrain) for min_f in self.time_frames])
        max_value_frame_description = abs(max([abs(self.time_frames[timing].feeling) for timing in self.time_frames]))

        for min_f in self.time_frames:
            self.time_frames[min_f].feeling_ktrain = self.time_frames[min_f].feeling_ktrain / max_value_frame_ktrain 

        for timing in self.time_frames:
            self.time_frames[timing].feeling = self.time_frames[timing].feeling / max_value_frame_description
        

          

    def show_descriptions_frames_with_caption(self, show_img=True):
        for frame_cap in self.frames_captions:
          if frame_cap.caption != None:
            frame_cap.caption.printC()
          else:
            print("No caption for this frame ... \n")
          frame_cap.frame_start.printF(show_img)
          
          if frame_cap.frame_end != None:
            frame_cap.frame_end.printF(show_img)

          print("\n#########################################\n")
    def calculate_positivity(self ,weightf1=2, weightf2=2, weightc=6, bias=-5):
      if (self.has_caption):
        numberCaption = 0
        for frames_cap in self.frames_captions:
          if frames_cap.caption != None:
            print("number, of caption : ", numberCaption)
            print("frames_cap.caption : ", frames_cap.caption.caption)
            print("frames_cap.caption feeling : ", frames_cap.caption.feeling)
            print("frames_cap.caption start time: ", frames_cap.caption.start)

            print("positivity : ", sigmoid(weightf1 * frames_cap.frame_start.feeling +
                                          weightf2 * frames_cap.frame_end.feeling + 
                                          weightc * frames_cap.caption.feeling +
                                          bias))
          else : 
              print("number of frame : ", numberCaption)
              print("timing start : ", frames_cap.frame_start.time)
              print("positivity : ", sigmoid(weightf1 * frames_cap.frame_start.feeling + bias))
          print("\n##########################################\n")

          numberCaption += 1
      else:
        numberFrame = 0
        for time_of_frame in self.time_frames:
          print("number of frame : ", numberFrame)
          print("positivity : ", sigmoid(self.time_frames[time_of_frame].feeling))
          print("\n############################################\n")
          numberFrame += 1
    def sum_the_positivity_of_vid(self, weightc=0.6, weightf=0.2, apart=False):
      weightfd= 1 - (weightc + weightf)
      
      somaF = 0
      somaFD = 0
      somaC = 0



      for time_of_frame in self.time_frames:        
        somaF += self.time_frames[time_of_frame].feeling_ktrain 
        somaFD += self.time_frames[time_of_frame].feeling
            
      
      if (not self.has_caption):
        weightfd = 1 - (weightf)
        return ((somaF * weightf) + (somaFD * weightfd))
      
      for cap in self.captions_save:
        somaC += cap.feeling
      
      if not apart:
        return (somaF * weightf) + (somaC * weightc) + (somaFD * weightfd)
      else:
        return (somaF , somaFD, somaC )
        

    # creates the frame_descriptio file
    def create_frame_description_file(self):
      if not os.path.isdir(self.video_dir + "/documents/"):
            os.mkdir(self.video_dir + "/documents/")
            print(f'Created the following directory: {self.video_dir}/documents/')
     
      frame_file_txt = self.video_dir + "/documents/" + self.video_name + "_frame.txt"
      file_with_frame_description = ""
      counter_frames = 1
      
      for frames_and_time in self.time_frames:
        file_with_frame_description += str(counter_frames) + "\n"
        file_with_frame_description += seconds_to_time(frames_and_time) + "\n"
        file_with_frame_description += "path : " + self.time_frames[frames_and_time].path + "\n"
        file_with_frame_description += "description : " + self.time_frames[frames_and_time].description + "\n\n"
        counter_frames += 1
      
      print("saving the file : ", frame_file_txt)
      with open(frame_file_txt, "w+") as new_file_frame:
        new_file_frame.write(file_with_frame_description)

      return file_with_frame_description

    def create_caption_description_file(self):
      if self.has_caption:
        if not os.path.isdir(self.video_dir + "/documents/"):
            os.mkdir(self.video_dir + "/documents/")
            print(f'Created the following directory: {self.video_dir}/documents/')


        caption_file_txt =   self.video_dir + "/documents/" + self.video_name + ".txt" 

        with open(caption_file_txt, "w+") as new_file:
            new_file.write(self.caption_str)


    # display a graph of the time siries    
    # returns a list of sentiments given a moment
    def get_intervale_sentiment_caption(self, moment):
      sentiments = []
      for cap in self.captions_save:
        if cap.start <= moment <= cap.end:
            sentiments.append(cap.feeling)
        if cap.start > moment:
          return sentiments
      return sentiments


    # applies the function in a set of sentiments in a espefic time
    # and plots, for example I'm using the average function, which takes a list and return a the average of the list
    # pCap the part of the caption that we will considere
    # pFrame the part of the frame that we will considere
    def create_time_siries(self, weightc=0.6, weightf=0.2):
      weightfd = 1 - weightf
      if (self.has_caption):
          weightfd = 1 - ( weightc + weightf )
      
      rc('figure', figsize=(20, 10))
      xC = []
      yC = []

      xF = []
      yF = []
      yFD = []
      yT = []


      last = 0.0
      lastD = 0.0

      start_min_caption = 0
      end_max_caption = 0
      start_min_frame = list(self.time_frames.keys())[0]
      end_max_frame = list(self.time_frames.keys())[-1]

      if self.has_caption:
        # since the list is order by the start time I can ensure that the min start will be the first 
        start_min_caption = self.captions_save[0].start
        # but I can't say the same for the max end
        end_max_caption = max([cap.end for cap in self.captions_save])

        start_min = min(start_min_caption, start_min_frame)
        min_aux = start_min
        end_max = max(end_max_caption, end_max_frame)

        while start_min <= end_max:
          xC.append(start_min)
          yC.append( average( self.get_intervale_sentiment_caption(start_min)))
          if start_min in self.time_frames:
                lastD = self.time_frames[start_min].feeling
                last = self.time_frames[start_min].feeling_ktrain
          xF.append(start_min)
          yF.append(last)
          yFD.append(lastD)
          start_min += 1
          # defines the porcentage of the caption/ frame 
          yT.append((yC[-1] * weightc ) + (yF[-1] * weightf) + (yFD[-1] * weightfd) )
        plt.plot(xC, yC, label="caption")
        plt.plot(xF, yF, label="frame")
        plt.plot(xF, yFD, label="frame d")
        plt.plot(xC, yT, label=  str(weightc * 100) + " cap " + str(weightf * 100)  + " frame" + str(weightfd * 100)  + " frame d")
        
        plt.xticks(np.arange(min_aux, end_max + 1, 10.0))      

      else: 
        last = 0.0
        while start_min_frame <= end_max_frame:
          if start_min_frame in self.time_frames:
            lastD = self.time_frames[start_min].feeling
            last = self.time_frames[start_min].feeling_ktrain
          xF.append(start_min_frame)
          yF.append(last)
          yFD.append(lastD)
          yT.append( (yF[-1] * weightf) + (yFD[-1] * weightfd) )
          
          start_min_frame += 1
        
        plt.plot(xF, yF, label="frame")
        plt.plot(xF, yFD, label="frame d")
        plt.plot(xC, yT, label=  str(weightf * 100)  + " frame" + str(weightfd * 100)  + " frame d")


        plt.xticks(np.arange(min(xF), max(xF)+1, 10.0))      

      
      plt.xlabel('time')
      plt.ylabel('feeling polarity')
      plt.title('time x feeling polarity')

      plt.legend()
      plt.show()
    def plot_sentiment_cap(self, time_start, time_end, value, counter):
      aux_value = time_start
      x = []
      y = []

      while aux_value <= time_end:
        x.append(aux_value)
        y.append(value)
        aux_value += 0.5
      plt.plot(x, y, label="caption" + str(counter))

    def plot_caption_sentiment_during_time(self, bool_frame=True):
      counter = 0
      rc('figure', figsize=(20, 10))
      
      if self.has_caption:
          for cap in self.captions_save:
            self.plot_sentiment_cap(cap.start, cap.end, cap.feeling, counter)
            counter += 1

        
      x = []
      yF = []
      yD = []
      if bool_frame:
        for timing in self.time_frames:
          x.append(timing)
          yF.append(self.time_frames[timing].feeling_ktrain)
          yD.append(self.time_frames[timing].feeling)

        plt.plot(x, yF, 'rs',label="frame")
        plt.plot(x, yD, 'b.', label= "frame D")
      if bool_frame or self.has_caption:
        plt.xlabel('time')
        plt.ylabel('feeling')
        plt.title('time x feelling')
        plt.show()