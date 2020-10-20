

### [Overview](#overview) | [Tutorials](#tutorials) | [Examples](#examples)  | [Installation](#installation)

<p align="center">
<img src="https://github.com/Vakihito/SentimentYoutube/blob/main/Logo.png?raw=true" width="100"/>
</p>


# Welcome to SentimentYoutube

### News and Announcements
- **2020-10-20**
    - First release of SentimentYoutube ! Version **0.0.1** cames with a simple way to analyze sentiment through out the video, it's not working properly, but defines the start of project by analysing both the frame and the caption of the video.

----

### Overview
**SentimentYoutube** is library makes easy way evaluate the sentiment in a frame, by using TensorFlow Keras to judge the feeling in a frame in positive or negative. In order to do this we analyze the frames of the video, also the description of generated by a deep learning algorithm and also the caption of the video.
This library uses Ktrain in order to train models and load models, and also uses pythia to describe a frame.

- In order to use this library you should learn abour it classes :
    - 'PythiaDemo'  :
        - **PythiaDemon()** - The main utility of this class is to describe a given image, this class is called in Utils to make the process easier. 
    - 'FrameExtractor' :
        - **FrameExtractor()** - The main utility of this library, it's the class the alows to analise a given video via it's ID.
        - **do_preparation()** - This funcion preprers everything needed before using the FrameEXtractor class. 
        - **sum_the_positivity_of_vid()** - This funcion returns the posivitity of the video
        - **show_descriptions_frames_with_caption()** - This function returns the description of each frame and also it's caption
        - **extract_obj.create_caption_description_file()** - This function creates a file with the captions of the video 
        - **extract_obj.create_frame_description_file()** - This function creates a file with the description of each frame. 
        - **extract_obj.plot_caption_sentiment_during_time()** - This function plots the feeling of the captions, frames and frames desptions over time.
        - **extract_obj.plot_caption_sentiment_during_time()** - This function plots the feeling of the captions, frames and frames desptions over time in a line. And also the weighted average of the 3 parameters.
    - 'Utils' :
        - This module has bunch of usefull functions that allow a easier use of the library
        - **predic_face()** - This function receives a path to a images and returns the positivity of a image.
        - **predic_text()** - This function receives a path to a images and returns the positivity of the sentence using a pre-trained model of the MDB dataset
        - **show_image_url** - Shows a image given a url
        - **show_image** - Shows a image given a path
        - **show_prediction** - Shows pythia predicion of a image given by either a path or an url

---

### Tutorials
Please see the following tutorial notebooks for a guide on how to use **SentimentYoutube** on your projects:
    
- **Tutorial 1**: [Introduction](https://colab.research.google.com/drive/1fZRv3pJA1Ie4H18zQugmAkDZdZ1usoHv?authuser=1#scrollTo=U4CASaopTpK-)

---
### Examples
Please see the following examples notebooks for a guide on how to use **SentimentYoutube** on your projects:
    
- **How to use** : [Introduction](https://colab.research.google.com/drive/1fZRv3pJA1Ie4H18zQugmAkDZdZ1usoHv?authuser=1#scrollTo=U4CASaopTpK-)

---

### Installation
This whole project was made in Google Colab, so Please use Google Colab nootebook in order to make this operantion work, other wise you may check our install.sh an change the paths to your liking, but I wouldn't suggest doing so...
- First of all make sure that you are using *GPU* enviroment.
- Than open the file content : <code> cd /content </code>
- Clone the git repository: <code> !git clone https://github.com/Vakihito/SentimentYoutube.git </code> 
- Open the file SentimentYoutube : <code> cd SentimentYoutube/ </code> 
- Install packages requirements: <code> !pip3 install -r requirements.txt </code> 
- Install enviroment requirements: <code> ./install.sh </code> 
- Thats all, now you are good to go !!!
