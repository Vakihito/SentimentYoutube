from .utils import predict_text
class caption_struct():
    '''
        saves the caption as a struct with start, as a int and end as a int 
        and the caption as a string
    '''
    def __init__(self, startC, endC, stringC, use_bert=True, neutral_range=0.1):
      
      

      # neutral_range - defines the range in wich the feeling will be classified as neutral
      self.start =  startC  
      self.end = endC
      self.caption = stringC
      
      self.use_bert = use_bert
      self.feeling_ktrain_text = predict_text(self.caption,method="IMDB",neutral_range=neutral_range) 
      if use_bert:
        (aux_nearst_query, self.feeling_bert) = predict_text(self.caption,method="Bert",only_sentiment=False,neutral_range=neutral_range) 
        self.feeling = (self.feeling_ktrain_text + self.feeling_bert)/2
        self.nearst_query = aux_nearst_query.loc[0] 
      else:
        self.feeling = self.feeling_ktrain_text

    def printC(self):
      print("Time caption Start : ", self.start)
      print("Time caption Ends : ", self.end)
      print("Caption itself : ", self.caption)
      print("Feeling AVG :", self.feeling)
      if self.use_bert:
        print("Feeling text ktrain : ", self.feeling_ktrain_text)
        print("Feeling text bert : ", self.feeling_bert)
        print("Nearst query lang : ", self.nearst_query["lang"])
        print("Nearst query string : ", self.nearst_query["word"])
        print("Nearst query value : ", self.nearst_query["query"])
        print("\n")