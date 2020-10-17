class caption_struct():
    '''
        saves the caption as a struct with start, as a int and end as a int 
        and the caption as a string
    '''
    def __init__(self, startC, endC, stringC):
      self.start =  startC  
      self.end = endC
      self.caption = stringC

    #   self.feeling = TextBlob(self.caption).sentiment # has atributes polarity and subjectivity
      
      prob_ktrain = predictor_text.predict(self.caption, return_proba=True)
      self.feeling = (prob_ktrain[1] - prob_ktrain[0]) 
      

    def printC(self):
      print("time caption Start : ", self.start)
      print("time caption Ends : ", self.end)
      print("caption itself : ", self.caption)
    #   print("feeling of the caption polaraty : ", self.feeling)
    #   print("feeling of the caption subjectivity : ", self.feeling.subjectivity)
      print("feeling text ktrain : ", self.feeling)
      print("\n")