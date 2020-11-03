from .utils import *
from PIL import Image
from IPython.display import display


class frame_struct():
    '''
      saves the frame time in seconds (int), 
      path (str),
      description (str),
      felling (textBlob.sentiment),
    '''
    def __init__(self, timeF, pathF, desciptionF, probf=0.85):
      self.time = timeF
      self.path = pathF
      # keep the list of faces in a list
      (self.faces, self.face_prob) = prop_of_having_face(pathF,confidence_arg=probf,save_img=True)
      self.description = desciptionF
      prob_ktrain = predictor_text.predict(self.description, return_proba=True)
      self.feeling = prob_ktrain[1] - prob_ktrain[0]  
      
      self.feeling_ktrain = 0
      
      # if the list is not empty, iterate over each face 
      # predicts the feeling in each face
      # then saves the avg feeling
      for faces_path in self.faces:
          self.feeling_ktrain += predict_face(faces_path)
      if self.faces:
          self.feeling_ktrain = self.feeling_ktrain/len(self.faces)
    
    def show_img(self):
      img = Image.open(self.path)
      img.thumbnail((256,256), Image.ANTIALIAS)
      display(img)
      return img

    def printF(self,show_i=True,show_c=True):
      print("time : ", self.time)
      print("path : ", self.path)
      if (show_i):
        self.show_img()
      print("description : ", self.description)
      print("feeling in description : ", self.feeling)
      if (show_c):
          for i in self.faces:
              show_image(i)
      print("probability of having a face : ", self.face_prob)
      print("feeling in frame image analasys: ", self.feeling_ktrain)