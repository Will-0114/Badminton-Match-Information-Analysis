from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap,QIcon
from PyQt5.QtCore import QTimer ,Qt
from PyQt5.QtWidgets import QFileDialog
from opencv_engine import opencv_engine,template_matching
from yolov5_engine import yolov5_engine

import time
import cv2

# videoplayer_state_dict = {
#  "stop":0,   
#  "play":1,
#  "pause":2     
# }

class video_controller(object):
    def __init__(self, video_path, ui,model_name = "",sender=None):
        self.video_path = video_path
        self.ui = ui
        self.sender = sender
        self.qpixmap_fix_width = 1280 # 16x9 = 1920x1080 = 1280x720 = 800x450
        self.qpixmap_fix_height = 720
        self.current_frame_no = 0
        self.videoplayer_state = "stop"
        self.model_name = model_name
        self.method = 0
        self.model = None
        self.OK = False
        print(self.model_name)
        if self.model_name != "":
            self.load_model()
        self.init_video_info()
        self.set_video_player()
        self.stateChaged()

        
    def fresh_model(self,model_name):
        self.model_name = model_name

        
    def load_model(self):
        self.model = yolov5_engine.load_model(self.model_name)
        print("done")
        
 

        




        

        



        
        
        


    # def pause(self):
    #     self.videoplayer_state = "pause"


            
