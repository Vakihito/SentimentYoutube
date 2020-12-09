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
    def __init__(self, timeF,  pathF, faces_P,neutral_range=0.05):
      # neutral_range - defines the range in wich the feeling on the frame description will be classified as neutral
      self.time = timeF
      self.path = pathF
      self.description = show_prediction(pathF)
 
      self.feeling_ktrain_text = predict_text(self.description,method="IMDB",neutral_range=neutral_range) 
      (aux_nearst_query, self.feeling_bert) = predict_text(self.description,method="Bert",only_sentiment=False,neutral_range=neutral_range) 
      self.feeling = (self.feeling_ktrain_text + self.feeling_bert)/2
      self.nearst_query = aux_nearst_query.loc[0] 
      
      self.feeling_ktrain = 0
      self.feeling_ktrain_avg = 0
      self.faces = faces_P
      self.n_faces = len(self.faces)
      
      

     
      lista_aux = []
      for faces_path in self.faces:
        lista_aux.append(predict_face(faces_path))
      
      if self.faces:
        self.feeling_ktrain_avg = sum(lista_aux)/self.n_faces
        self.feeling_ktrain = consent_values(lista_aux)/self.n_faces
      

        

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
      print("Feeling AVG :", self.feeling)
      print("Feeling text ktrain : ", self.feeling_ktrain_text)
      print("Feeling text bert : ", self.feeling_bert)
      print("Nearst query lang : ", self.nearst_query["lang"])
      print("Nearst query string : ", self.nearst_query["word"])
      print("Nearst query value : ", self.nearst_query["query"])
      if (show_c):
          for i in self.faces:
              print(i)
              show_image(i)
      print("Number_of_faces analised : ", self.n_faces)
      print("Feeling in faces analasys: ", self.feeling_ktrain)
      print("Average feeling in faces analasys: ", self.feeling_ktrain_avg)