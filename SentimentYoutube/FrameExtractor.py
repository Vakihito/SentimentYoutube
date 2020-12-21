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
    def __init__(self, video_name, video_id, video_length, start=0, end=None ,use_bert=False, use_ktrain=True,video_path="/content/vqa-maskrcnn-benchmark/", video_dir="./dataset/", frames_frequency=200):
        self.video_name = safe_filename(video_name)
        self.video_id = video_id
        self.video_path = video_path + self.video_name + ".mp4"
        self.video_dir = video_dir + self.video_name  + self.video_id + "_dir"
        self.video_len = video_length
        self.img_name = self.video_name + "_img"

        if end == None:
            self.start_time = start
            self.end_time = self.video_len
        else :
            self.start_time = start
            self.end_time = end

        self.use_bert = use_bert
        self.use_ktrain = use_ktrain

        self.frames_frequency = frames_frequency
        self.vid_cap = cv2.VideoCapture(self.video_path)
        self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

        self.all_images = [] #saves all imges of the video
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
            if (line_counter + 1) % 4  == 0 and self.video_len >= time_e:  
                string_cap = line
                self.captions_save.append(caption_struct(time_s, time_e, string_cap, use_bert=self.use_bert, use_ktrain=self.use_ktrain))
                self.key_time.append(time_s)
                self.key_time.append(time_e)
            line_counter += 1

        self.key_time = sorted(set(self.key_time))




    def _list_faces_neighbors(self, list_faces):
        list_r = []
        for faces in list_faces:
            # print(faces)
            if isinstance(faces, list)  and len(faces) > 1 and faces[1] != -1:
                list_r += faces[1:]
        return list_r
    def _create_img(self, img, counter_img, img_ext='.jpg', create=True):
        img_path = os.path.join(self.video_dir, ''.join([self.img_name, '_', str(counter_img), img_ext]))
        print("image path: ", img_path)
        cv2.imwrite(img_path, img)
        self.all_images.append(img_path)
        return img_path
    

        

    def extract_frames_no_caption(self, img_ext='.jpg',prob_f=0.8,by_frame=False):
        # print(self.key_time)
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
        
        if self.video_dir is None:
            self.video_dir = os.getcwd()
        else:
            if not os.path.isdir(self.video_dir):
                os.mkdir(self.video_dir)
                print(f'Created the following directory: {self.video_dir}')
        
        caption_counter = 0
        last_time_added = -1
        # counts the number of frames
        frame_cnt = 0
        # counts the number of images saved
        img_cnt = 0
        # number of neighbor needed to verify the faces ( conts the current frame)
        n_neighbors = 21
        mid_value = int(n_neighbors/2)
        
        # this list is a list of lists, that holds path of a img and the faces related to the path of the img
        # so all the list inside this list will be like this : [pathF, img_face_crop0 , img_face_crop1 , img_face_crop2, img_face_crop3 ...] 
        list_faces = [-1 for i in range(n_neighbors)]

        # this is a numpy list that holds number of frames that we will have to wait until we save the frame
        list_idx_frames = np.array([]) 
        counter_aux = 0

        


        while self.vid_cap.isOpened():
            
            # update the list_faces
            cur_second = int((frame_cnt / self.fps))
            success,image = self.vid_cap.read() 

            if not success:
                break
            # print("cur_time : ",cur_second)
            # updates the first elemento of the list
            shift(list_faces);
            # makes a rotation right
            # update a our list 
            list_faces[0] = [[image, cur_second]]
            
            if (cur_second > self.end_time):
                return
            if (cur_second >= self.start_time):
                if (counter_aux % self.frames_frequency == 0 and last_time_added != list_faces[0][0][1]):
                    list_idx_frames = np.append(list_idx_frames,[mid_value + 1])

                if (len(list_idx_frames) > 0):
                    # iterating over backwards to since we need to maintain the order of the frames
                    # sorted by descreasing time 
                    for i in range(len(list_faces) - 1, -1,-1):
                        #saving each frame if was not saved
                        # so the firs element of a list of a  list_faces, will be [image, cur sencond, ]
                        # follow by some faces_images

                        # checks if the element was already processed ( the img was writen )
                        # if not writes the image 
                        if isinstance(list_faces[i], list) and len(list_faces[i][0]) < 3:
                            
                            path_img = self._create_img(list_faces[i][0][0], img_cnt, img_ext)
                            list_faces[i][0] +=  [path_img]

                            img_cnt += 1
                            
                            # checking if was not processed (faces in te frames)
                            # so list_faces[i] will be [[image, cur sencond, path]] if we din't check for faces in the past
                            (list_aux1, _) = prop_of_having_face(list_faces[i][0][2],confidence_arg=prob_f,save_img=True)
                            list_faces[i] += list_aux1
                            
                        
                    # update the list of frames
                    list_idx_frames -= 1

                    

                    # checks if the value is possible to create 
                    if (list_idx_frames[0] == 0 ):
                        if (counter_aux >= mid_value):
                            self.time_frames[list_faces[mid_value][0][1]] = frame_struct(list_faces[mid_value][0][1],
                                                                                            list_faces[mid_value][0][2],
                                                                                            self._list_faces_neighbors(list_faces),
                                                                                            use_bert=self.use_bert,
                                                                                            use_ktrain=self.use_ktrain)
                        else :
                            print(counter_aux)
                            self.time_frames[list_faces[counter_aux][0][1]] = frame_struct(list_faces[counter_aux][0][1],
                                                                                            list_faces[counter_aux][0][2],
                                                                                            self._list_faces_neighbors(list_faces),
                                                                                            use_bert=self.use_bert,
                                                                                            use_ktrain=self.use_ktrain)
                        list_idx_frames = np.delete(list_idx_frames,0)
                        
                counter_aux += 1
            frame_cnt += 1
        self.vid_cap.release()
        cv2.destroyAllWindows()
    
        while len(list_idx_frames) :
        
            shift(list_faces);
            # iterating over backwards to since we need to maintain the order of the frames
            # sorted by descreasing time 
            for i in range(len(list_faces) - 1, -1,-1):
                #saving each frame if was not saved
                # so the firs element of a list of a  list_faces, will be [image, cur sencond, ]
                # follow by some faces_images

                # checks if the element was already processed (if the img was writen )
                # if not writes the image and 
                if isinstance(list_faces[i], list) and len(list_faces[i][0]) < 3:
                    # writes the img and returns the path
                    path_img = self._create_img(list_faces[i][0][0], img_cnt, img_ext)
                    list_faces[i][0] +=  [path_img]

                    img_cnt += 1
                    
                    # checking if was not processed (faces in te frames)
                    # so list_faces[i] will be [[image, cur sencond, path]], now we add the faces path 
                    (list_aux1, _) = prop_of_having_face(list_faces[i][0][2],confidence_arg=prob_f,save_img=True)
                    list_faces[i] += list_aux1

            
            
            self.time_frames[list_faces[mid_value][0][1]] = frame_struct(list_faces[mid_value][0][1],list_faces[mid_value][0][2], self._list_faces_neighbors(list_faces), use_bert=self.use_bert, use_ktrain=self.use_ktrain)
            list_idx_frames = np.delete(list_idx_frames,0)
            list_idx_frames -= 1


        
        
        
    def extract_frames_caption(self, img_ext='.jpg',prob_f=0.8,by_frame=False):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
        
        if self.video_dir is None:
            self.video_dir = os.getcwd()
        else:
            if not os.path.isdir(self.video_dir):
                os.mkdir(self.video_dir)
                print(f'Created the following directory: {self.video_dir}')
        
        caption_counter = 0
        while (self.start_time > self.captions_save[caption_counter].start):
            caption_counter += 1  
        
            

        last_time_added = -1
        # counts the number of frames
        frame_cnt = 0
        # counts the number of images saved
        img_cnt = 0
        # number of neighbor needed to verify the faces ( conts the current frame)
        n_neighbors = 21
        mid_value = int(n_neighbors/2)
        
        # this list is a list of lists, that holds path of a img and the faces related to the path of the img
        # so all the list inside this list will be like this : [pathF, img_face_crop0 , img_face_crop1 , img_face_crop2, img_face_crop3 ...] 
        list_faces = [-1 for i in range(n_neighbors)]

        # this is a numpy list that holds number of frames that we will have to wait until we save the frame
        list_idx_frames = np.array([]) 
        counter_aux = 0
        bool_possible_key = False
        


        while self.vid_cap.isOpened():
            
            # update the list_faces
            cur_second = int((frame_cnt / self.fps))
            

            success,image = self.vid_cap.read() 

            if not success:
                break
            
            

            # updates the first elemento of the list
            shift(list_faces);
            # makes a rotation right
            # update a our list 
            list_faces[0] = [[image, cur_second]]

            if (cur_second > self.end_time):
                return
            if (cur_second >= self.start_time):
                
                bool_possible_key = (len(self.key_time) > caption_counter and
                                    cur_second ==  self.key_time[caption_counter])
                # checking if the current time is  a key time
                if (bool_possible_key ):
                    last_time_added = list_faces[0][0][1]
                    list_idx_frames = np.append(list_idx_frames,[mid_value + 1])
                    caption_counter += 1
                elif (counter_aux % self.frames_frequency == 0 and last_time_added != list_faces[0][0][1]):
                    list_idx_frames = np.append(list_idx_frames,[mid_value + 1])

                if (len(list_idx_frames) > 0):
                    # iterating over backwards to since we need to maintain the order of the frames
                    # sorted by descreasing time 
                    for i in range(len(list_faces) - 1, -1,-1):
                        #saving each frame if was not saved
                        # so the firs element of a list of a  list_faces, will be [image, cur sencond, ]
                        # follow by some faces_images

                        # checks if the element was already processed ( the img was writen )
                        # if not writes the image 
                        if isinstance(list_faces[i], list) and len(list_faces[i][0]) < 3:
                            
                            path_img = self._create_img(list_faces[i][0][0], img_cnt, img_ext)
                            list_faces[i][0] +=  [path_img]

                            img_cnt += 1
                            
                            # checking if was not processed (faces in te frames)
                            # so list_faces[i] will be [[image, cur sencond, path]] if we din't check for faces in the past
                            (list_aux1, _) = prop_of_having_face(list_faces[i][0][2],confidence_arg=prob_f,save_img=True)
                            list_faces[i] += list_aux1
                            
                        
                    # update the list of frames
                    list_idx_frames -= 1

                    

                    # checks if the value is possible to create 
                    if (list_idx_frames[0] == 0 ):
                        if (counter_aux >= mid_value):
                            self.time_frames[list_faces[mid_value][0][1]] = frame_struct(list_faces[mid_value][0][1],
                                                                                            list_faces[mid_value][0][2],
                                                                                            self._list_faces_neighbors(list_faces),
                                                                                            use_bert=self.use_bert,
                                                                                            use_ktrain=self.use_ktrain)
                        else :
                            print(counter_aux)
                            self.time_frames[list_faces[counter_aux][0][1]] = frame_struct(list_faces[counter_aux][0][1],
                                                                                            list_faces[counter_aux][0][2],
                                                                                            self._list_faces_neighbors(list_faces),
                                                                                            use_bert=self.use_bert,
                                                                                            use_ktrain=self.use_ktrain)
                        list_idx_frames = np.delete(list_idx_frames,0)
                        
                counter_aux += 1
            frame_cnt += 1
        self.vid_cap.release()
        cv2.destroyAllWindows()
      
        while len(list_idx_frames) :
        
            shift(list_faces);
            
            # iterating over backwards to since we need to maintain the order of the frames
            # sorted by descreasing time 
            for i in range(len(list_faces) - 1, -1,-1):
                #saving each frame if was not saved
                # so the firs element of a list of a  list_faces, will be [image, cur sencond, ]
                # follow by some faces_images

                # checks if the element was already processed (if the img was writen )
                # if not writes the image and 
                if isinstance(list_faces[i], list) and len(list_faces[i][0]) < 3:
                    # writes the img and returns the path
                    path_img = self._create_img(list_faces[i][0][0], img_cnt, img_ext)
                    list_faces[i][0] +=  [path_img]

                    img_cnt += 1
                    
                    # checking if was not processed (faces in te frames)
                    # so list_faces[i] will be [[image, cur sencond, path]], now we add the faces path 
                    (list_aux1, _) = prop_of_having_face(list_faces[i][0][2],confidence_arg=prob_f,save_img=True)
                    list_faces[i] += list_aux1

            
            
            self.time_frames[list_faces[mid_value][0][1]] = frame_struct(list_faces[mid_value][0][1],list_faces[mid_value][0][2], self._list_faces_neighbors(list_faces), use_bert=self.use_bert, use_ktrain=self.use_ktrain)
            list_idx_frames = np.delete(list_idx_frames,0)
            list_idx_frames -= 1
        

    def correlates_caption_frame(self):
      print(self.time_frames.keys())
      for cap in self.captions_save:
        if cap.start >= self.start_time and cap.end <= self.end_time:
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
            if(max_value_caption_ktrain == 0):
                max_value_caption_ktrain=1
            for i in range(len(self.captions_save)):
                self.captions_save[i].feeling = self.captions_save[i].feeling / max_value_caption_ktrain

        max_value_frame_ktrain = max([abs(self.time_frames[min_f].feeling_ktrain) for min_f in self.time_frames])
        if (max_value_frame_ktrain == 0):
            max_value_frame_ktrain=1
        max_value_frame_description = abs(max([abs(self.time_frames[timing].feeling) for timing in self.time_frames]))
        if (max_value_frame_description == 0):
            max_value_frame_description=1

        max_value_frame_ktrain_avg = abs(max([abs(self.time_frames[timing].feeling_ktrain_avg) for timing in self.time_frames]))
        if (max_value_frame_ktrain_avg == 0):
            max_value_frame_ktrain_avg=1

        for min_f in self.time_frames:
            self.time_frames[min_f].feeling_ktrain = self.time_frames[min_f].feeling_ktrain / max_value_frame_ktrain
            self.time_frames[min_f].feeling = self.time_frames[min_f].feeling / max_value_frame_description
            self.time_frames[min_f].feeling_ktrain_avg = self.time_frames[min_f].feeling_ktrain_avg / max_value_frame_ktrain_avg



          

    def show_descriptions_frames_with_caption(self, show_img=True):
        if (not self.has_caption ):
            self.show_descriptions_from_all_frames()

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


    def sum_the_positivity_of_vid_consent(self):
      xC = []
      xF = []
      yC = []
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
          yT.append( consent_values( [yC[-1], yF[-1], yFD[-1]] ) )  

      else: 
        last = 0.0
        while start_min_frame <= end_max_frame:
          if start_min_frame in self.time_frames:
            lastD = self.time_frames[start_min_frame].feeling
            last = self.time_frames[start_min_frame].feeling_ktrain
          xF.append(start_min_frame)
          yF.append(last)
          yFD.append(lastD) 
          yT.append( consent_values( [yF[-1], yFD[-1]] ) )
          start_min_frame += 1
      
      return sum(yT)

    def sum_the_positivity_of_vid(self, weightc=0.6, weightf=0.2, apart=False,consent=True):

      if consent:
          return self.sum_the_positivity_of_vid_consent()

      weightfd= 1 - (weightc + weightf)
      
      somaF = 0
      somaFD = 0
      somaC = 0

      somaT = 0


      for time_of_frame in self.time_frames:        
        somaF += self.time_frames[time_of_frame].feeling_ktrain 
        somaFD += self.time_frames[time_of_frame].feeling
        
            
      
      if (not self.has_caption):
        weightfd = 1 - (weightf)
        if not apart:
            return ((somaF * weightf) + (somaFD * weightfd))
        return (somaF, somaFD)
      
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
    # consent defines the classifications diverge returns neutral.
    def create_time_siries(self, start_plt_time=0.0, end_plt_time=None, weightc=0.6, weightf=0.2, consent_F=True ,consent=True):
      weightfd = 1 - weightf
      if (self.has_caption):
          weightfd = 1 - ( weightc + weightf )

      rc('figure', figsize=(20, 10))
      xC = []
      yC = []

      xF = []
      yF = []
      yFD = []
      yFIMDB = []
      yFBert = []
      yT = []

      def_time_start = max(self.start_time, start_plt_time)
      def_time_end = self.end_time
      if end_plt_time != None: 
        def_time_end = min(self.end_time, end_plt_time) 
      if start_plt_time >= def_time_end:
          return False

      last = 0.0
      lastD = 0.0
      lastFIMDB = 0.0
      lastFBert = 0.0

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
        if start_min < def_time_start:
          start_min = def_time_start
        
        min_aux = start_min
        end_max = max(end_max_caption, end_max_frame, def_time_end)
        if end_max > def_time_end:
          end_max = def_time_end

        while start_min <= end_max:
          xC.append(start_min)
          yC.append( average( self.get_intervale_sentiment_caption(start_min)))
          if start_min in self.time_frames:
            lastD = self.time_frames[start_min].feeling
            if self.use_ktrain:
                lastFIMDB = self.time_frames[start_min].feeling_ktrain_text
            if self.use_bert:
                lastFBert = self.time_frames[start_min].feeling_bert
            if (consent_F):
                last = self.time_frames[start_min].feeling_ktrain
            else :
                last = self.time_frames[start_min].feeling_ktrain_avg


          xF.append(start_min)
          yF.append(last)
          yFD.append(lastD)
          if self.use_ktrain:
            yFIMDB.append(lastFIMDB)
          if self.use_bert:
            yFBert.append(lastFBert)
          
          start_min += 1
          # defines the porcentage of the caption/ frame 
          if (consent):
            yT.append( consent_values( [yC[-1], yF[-1], yFD[-1]] ) )
          else:
            yT.append((yC[-1] * weightc ) + (yF[-1] * weightf) + (yFD[-1] * weightfd) )

        plt.plot(xC, yC, label="caption")
        plt.plot(xF, yF, label="frame")
        plt.plot(xF, yFD, label="frame d")
        
        if self.use_ktrain:
            plt.plot(xF, yFIMDB, label="frame IMDB")
        
        if self.use_bert:
            plt.plot(xF, yFBert, label="frame Bert")

        if (consent):
          plt.plot(xC, yT, label=  "consent between the caption feeling, frame descriptio and the face analysis ")
        else:
          plt.plot(xC, yT, label=  str(weightc * 100) + " cap " + str(weightf * 100)  + " frame" + str(weightfd * 100)  + " frame d")
        
        plt.xticks(np.arange(min_aux, end_max + 1, 10.0))      

      else: 
        last = 0.0
        if end_max_frame > def_time_end:
          end_max_frame = def_time_end

        if start_min_frame < def_time_start:
          start_min_frame = def_time_start

        while start_min_frame <= end_max_frame:
          if start_min_frame in self.time_frames:
            lastD = self.time_frames[start_min_frame].feeling
            if self.use_ktrain:
                lastFIMDB = self.time_frames[start_min_frame].feeling_ktrain_text
            if self.use_bert:
                lastFBert = self.time_frames[start_min_frame].feeling_bert
            if (consent_F):
                last = self.time_frames[start_min_frame].feeling_ktrain
            else :
                last = self.time_frames[start_min_frame].feeling_ktrain_avg
          xF.append(start_min_frame)
          yF.append(last)
          yFD.append(lastD)
          
          if self.use_ktrain:
            yFIMDB.append(lastFIMDB)
          if self.use_bert:
            yFBert.append(lastFBert)

          if (consent):
            yT.append( consent_values( [yF[-1], yFD[-1]] ) )
          else:
            yT.append( (yF[-1] * weightf) + (yFD[-1] * weightfd))
          
          start_min_frame += 1
        
        plt.plot(xF, yF, label="frame")
        plt.plot(xF, yFD, label="frame d")

        if self.use_ktrain:
            plt.plot(xF, yFIMDB, label="frame IMDB")
        if self.use_bert:
            plt.plot(xF, yFBert, label="frame Bert")

        plt.plot(xF, yT, label=  str(weightf * 100)  + " frame" + str(weightfd * 100)  + " frame d")


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

    def plot_caption_sentiment_during_time(self, start_plt_time=0.0, end_plt_time=None,consent_F=True,bool_frame=True):

      counter = 0
      rc('figure', figsize=(20, 10))
      
      def_time_start = max(self.start_time, start_plt_time)
      def_time_end = self.end_time
      if end_plt_time != None: 
        def_time_end = min(self.end_time, end_plt_time) 
      if start_plt_time >= def_time_end:
          return False

      if self.has_caption:
          for cap in self.captions_save:
            if (cap.start >= def_time_start and cap.end <= def_time_end):
                self.plot_sentiment_cap(cap.start, cap.end, cap.feeling, counter)
                counter += 1

        
      x = []
      yF = []
      yD = []
      if bool_frame:
        for timing in self.time_frames:
          if (timing >= def_time_start and timing <= def_time_end):
            x.append(timing)
            if consent_F:
                yF.append(self.time_frames[timing].feeling_ktrain)
            else:
                yF.append(self.time_frames[timing].feeling_ktrain_avg)
            yD.append(self.time_frames[timing].feeling)

        plt.plot(x, yF, 'rs',label="frame")
        plt.plot(x, yD, 'b.', label= "frame D")
      if bool_frame or self.has_caption:
        plt.xlabel('time')
        plt.ylabel('feeling')
        plt.title('time x feelling')
        plt.show()
 


# does every preparation step before
def do_preparation(id, start_time=0, end_time=None, using_bert=True ,using_ktrain=True ,lang='en', frames_t=200):
  if frames_t < 10:
    print("\033[93merror, frame rate is less than minimum (10)\x1b[0m")
    return []
  
  ulr_to_download = 'https://www.youtube.com/watch?v=' + id
  #   itag = 18
  print("downloading the video from the ulr : ", ulr_to_download)
  video = YouTube(ulr_to_download)

  
  if  start_time == end_time or start_time < 0 :
    print("\033[91mError : invalid time range\033[0m")
    return 1

  if end_time == None:
    print("extracting from: ", start_time, "s to ", video.length, "s")
  else:
    if end_time > video.length or end_time < start_time:
        print("\033[91mError : invalid time range\033[0m")
        return 1
    print("extracting from: ", start_time, "s to ", end_time, "s")
    
  
  

  max_res = 0
  itag_max = -1
  for stream in video.streams.all():
      if stream.resolution and max_res < (int)(stream.resolution[:-1]) and stream.mime_type == "video/mp4" :
          max_res = (int)(stream.resolution[:-1])
          itag_max = stream.itag 
  video.streams.get_by_itag(str(itag_max)).download()

  if lang != 'en' and lang != 'a.en':
    extractor_obj = FrameExtractor(video.title,id,  start=start_time, end=end_time ,use_bert=is_using_bert(), use_ktrain=False,video_length=video.length,frames_frequency=frames_t)
  else:
    extractor_obj = FrameExtractor(video.title,id, start=start_time, end=end_time ,use_bert=is_using_bert(), use_ktrain=using_ktrain,video_length=video.length,frames_frequency=frames_t)

  print(video.captions.all())
  for cap in video.captions.all():
    if cap.code == lang:
      extractor_obj.has_caption = True

  extractor_obj.just_extract(video,lang)
  extractor_obj.normalize_sentiment();


  # removing the movie
  os.remove(extractor_obj.video_path)

  # enviando os dados e tornando-os visíveis no drive


  return extractor_obj      