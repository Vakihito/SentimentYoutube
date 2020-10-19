from .utils import predict_face, predict_text
from PIL import Image
from IPython.display import display


class frame_struct():
    '''
      saves the frame time in seconds (int), 
      path (str),
      description (str),
      felling (textBlob.sentiment),
    '''
    def __init__(self, timeF, pathF, desciptionF):
      self.time = timeF
      self.path = pathF
      self.description = desciptionF
      prob_ktrain = predict_text(self.description, return_proba=True)
      self.feeling = prob_ktrain[1] - prob_ktrain[0]  # has atributes polarity and subjectivity
      self.feeling_ktrain = predict_face(self.path)[0] - 1
    
    def show_img(self):
      img = Image.open(self.path)
      img.thumbnail((256,256), Image.ANTIALIAS)
      display(img)
      return img

    def printF(self,show_i=True):
      print("time : ", self.time)
      print("path : ", self.path)
      if (show_i):
        self.show_img()
      print("description : ", self.description)
      print("feeling in description : ", self.feeling)
      print("feeling in frame image analasys: ", self.feeling_ktrain)