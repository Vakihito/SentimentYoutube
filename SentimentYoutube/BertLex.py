from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
from tensorflow.keras.layers import Dense, Input, LSTM, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from numpy import dot
from numpy.linalg import norm
from tqdm.notebook import tqdm
import tensorflow as tf
import pickle



def get_bert_sent_features(inputs):
  m = tf.keras.Model(inputs=dnn.input,outputs=dnn.layers[0:1][0].output)
  return np.array(m(inputs))




def load_bert(DirPath):
    np.set_printoptions(threshold=100)
    logging.basicConfig(format='%(asctime)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO,handlers=[LoggingHandler()])

    global language_model, dnn, df_lex 
    language_model = SentenceTransformer(DirPath + '/language_model')


    dnn = Sequential()
    dnn.add(Dense(512, input_dim=512, activation='linear', name='lastlayer'))
    dnn.add(Dense(1))
    dnn.compile(loss='mse', optimizer='adam',metrics=['mse','mae'])
    dnn.summary()
    dnn.load_weights(DirPath + '/bert_lex_model/model.weights')

    with open(DirPath + 'df_lex.pickle', 'rb') as handle:
        df_lex = pickle.load(handle)
    



def get_bert_lex(text,lang='all',k=3, max_query=1.5):
  
  global df_lex

  # global
  features = np.array([list(language_model.encode([text],show_progress_bar=False)[0])])
  features = get_bert_sent_features(features)
  df_lex['query'] = np.linalg.norm(np.array(list(df_lex['bert_lex_sent']))-features,axis=1)

  temp =df_lex.sort_values(by='query').head(k)
  

  temp.reset_index(inplace=True)
  counter = 0
  for i in temp['query'].to_list():
      if i > max_query:
          break
      counter += 1


  sentiment = 0
  if counter == 0:
      temp = temp.loc[0:0]
  else:
      temp = temp.loc[0:counter]
      sentiment = np.mean(np.array(temp['sentiment'].to_list()))

  

  return temp, sentiment