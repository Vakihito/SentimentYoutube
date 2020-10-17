class frame_caption():
    '''
        saves the caption and the frames a list
        the first freme will be .frames[0], and the last will be .frames[1]
        the caption will be .caption.caption 
        the start time will be .caption.start
        the end will be .caption.end
    '''
    def __init__(self, frame_s, frame_e=None, caption=None):
        self.frame_start = frame_s # holds the first frame
        self.frame_end = frame_e  # holds the last frame
        self.caption = caption  # holds the caption of the frame
