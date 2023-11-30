from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap,QIcon
from PyQt5.QtWidgets import QFileDialog,QButtonGroup,QTableWidgetItem
from PyQt5.QtCore import QTimer ,Qt

from WorkThread import WorkThread
from opencv_engine import opencv_engine,template_matching
from yolov5_engine import yolov5_engine
from mediapipe_engine import mediapipe
from detect_pose import detect_pose,load_pose_weight
from mobilenet import mobilenet
from queue import Queue

import time
import cv2
import numpy as np


from view.UI import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self) 
        self.qpixmap_fix_width = 1280 # 16x9 = 1920x1080 = 1280x720 = 800x450
        self.qpixmap_fix_height = 720
        self.current_frame_no = 0
        self.videoplayer_state = "stop"
        self.method = 0
        self.model = None
        self.pose_model = None
        self.mobilenetmodel = None
        self.OK = False
        self.ClickEvent()
        self.stateChaged()
        self.buttonGroup()
        self.TM = template_matching()
        self.OE = opencv_engine()
        self.MN = mobilenet()
        a = [1,2,3,4,5,6,7]
        self.row = self.ui.tableWidget.rowCount()
        self.conf = round(self.ui.SpinBox_Conf.value(),2)
        self.Iou = round(self.ui.SpinBox_Iou.value(),2)
        self.ball_list = Queue()
        print(self.conf,self.Iou)
        
    def Iou_ValueChanged(self):
        self.Iou = round(self.ui.SpinBox_Iou.value(),2)

    def conf_ValueChaged(self):
        self.conf = round(self.ui.SpinBox_Conf.value(),2)
        
        
    def stateChaged(self):
        self.ui.radioButton_CCOEFF.toggled.connect(self.OnClicked)
        self.ui.radioButton_NORM_CCOEFF.toggled.connect(self.OnClicked)
        self.ui.radioButton_CCORR.toggled.connect(self.OnClicked)
        self.ui.radioButton_NORM_CCORR.toggled.connect(self.OnClicked)
        self.ui.radioButton_NORM_SQDIFF.toggled.connect(self.OnClicked)
        self.ui.radioButton_SQDIFF.toggled.connect(self.OnClicked)
        self.ui.SpinBox_Conf.valueChanged.connect(self.conf_ValueChaged)
        self.ui.SpinBox_Iou.valueChanged.connect(self.Iou_ValueChanged)
        
        
    def buttonGroup(self):
        self.btngroup_method = QButtonGroup()
        self.btngroup_method.addButton(self.ui.radioButton_CCOEFF)
        self.btngroup_method.addButton(self.ui.radioButton_NORM_CCOEFF)
        self.btngroup_method.addButton(self.ui.radioButton_CCORR)
        self.btngroup_method.addButton(self.ui.radioButton_NORM_CCORR)
        self.btngroup_method.addButton(self.ui.radioButton_NORM_SQDIFF)
        self.btngroup_method.addButton(self.ui.radioButton_SQDIFF)
        
    def OnClicked(self):
        
        method = {'CCOEFF':0, 'NORM_CCOEFF':1, 'CCORR':2,
                'NORM_CCORR':3, 'NORM_SQDIFF':5, 'SQDIFF':4}
        radioBtn = self.sender()
        if radioBtn.isChecked():
            
            self.method = method[radioBtn.text()]
            print(self.method)
            
            
        
    def ClickEvent(self):
        self.ui.button_openfile.clicked.connect(self.open_file)
        self.ui.button_openmodel.clicked.connect(self.open_model_file)
        self.ui.button_play.clicked.connect(self.play_and_pause) # connect to function()
        self.ui.button_stop.clicked.connect(self.stop)
        self.ui.button_openposemodel.clicked.connect(self.open_pose_model_file)
        self.ui.button_mobilenetmodel.clicked.connect(self.open_mobilenet_model_file)
        
    def play_and_pause(self):
        if self.videoplayer_state == "play":
            self.ui.button_play.setIcon(QIcon(QPixmap("./Icon/play.png")))
            self.videoplayer_state = "pause"
        else:
            self.ui.button_play.setIcon(QIcon(QPixmap("./Icon/pause.png")))
            self.videoplayer_state = "play"

    def stop(self):
        self.videoplayer_state = "stop"
        self.ui.button_play.setIcon(QIcon(QPixmap("./Icon/play.png")))
        
        
    def load_model_file_execute(self,filename):
        content = filename
        work = WorkThread(filename)
        work.start()
        work.signals.connect(self.load_model)
        
        
        
    def open_model_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Model Files(*.pt)") # start path        
        
        if not filename:
            return
        self.load_model_file_execute(filename)
        #self.model_name = filename
        # self.ui.label_model_name.setText(f"{filename}")
        # self.model = yolov5_engine.load_model(filename)  
    
    def open_pose_model_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Model Files(*.pth)") # start path        
        
        if not filename:
            return
        self.load_pose_model(filename)
        
    def open_mobilenet_model_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Model Files(*.pth)") # start path        
        
        if not filename:
            return
        self.load_mobilenet_model(filename)
        
    def load_mobilenet_model(self,filename):
        self.ui.label_mobilenet_model_name.setText(f"{filename}")
        self.mobilenet_model = self.MN.load_model(filename)
        
    def load_pose_model(self,filename):
        self.ui.label_pose_model_name.setText(f"{filename}")
        self.pose_model = load_pose_weight(filename)
    
    def load_model(self,filename):
        
        self.ui.label_model_name.setText(f"{filename}")
        self.model = yolov5_engine.load_model(filename,conf=self.conf,iou=self.Iou)  

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file Window", "./", "Video Files(*.mp4 *.avi)") # start path        
        
        if not filename:
            return
        
        self.video_path = filename
        self.ui.label_filepath.setText(f"{self.video_path}")
        self.init_video_info()
        self.set_video_player()
        
        (1,2,3)
        
    def init_video_info(self):
        videoinfo = self.OE.getvideoinfo(self.video_path)
        self.vc = videoinfo["vc"] 
        self.video_fps = videoinfo["fps"] 
        self.video_total_frame_count = videoinfo["frame_count"] 
        self.video_width = videoinfo["width"]
        self.video_height = videoinfo["height"] 
        self.ui.slider_videoframe.setRange(0, self.video_total_frame_count-1)
        self.ui.slider_videoframe.valueChanged.connect(self.getslidervalue)
        self.ui.slider_videoframe.sliderPressed.connect(self.sliderpress)
        self.ui.slider_videoframe.sliderReleased.connect(self.sliderrelease)
    
    def sliderpress(self):
        self.videoplayer_state = "pause"
        
    def sliderrelease(self):
        self.videoplayer_state = "play"
        
    def getslidervalue(self):
        self.current_frame_no = self.ui.slider_videoframe.value()
        
    def setslidervalue(self, value):
        self.ui.slider_videoframe.setValue(self.current_frame_no)
        
    def set_video_player(self):
        self.timer=QTimer() # init QTimer
        self.timer.timeout.connect(self.timer_timeout_job) # when timeout, do run one
        self.timer.start(1000//self.video_fps) # start Timer, here we set '1000ms//Nfps' while timeout one time
        self.timer.start(1) # but if CPU can not decode as fast as fps, we set 1 (need decode time)
        
    def load_info(self,player,score1,score2):
        col = 0
        
        if player == None:
            return
        self.ui.tableWidget.setRowCount(self.row+1)
        
        scores = []
        for i in range(3):
            scores += [str(score1[i]) +" : " +str(score2[i])]
        #print(scores)
        info = [self.current_time,player[0],player[1],scores[0],scores[1],scores[2]]
                
        for item in info:
            cell = QTableWidgetItem(str(item))
            self.ui.tableWidget.setItem(self.row,col,cell)
            col+=1
        self.row+=1
        
    def set_current_frame_no(self, frame_no):
        #print("x")
        self.vc.set(1, frame_no)
        
    def __get_next_frame(self):
        ret, frame = self.vc.read()
        hours_all = self.video_total_frame_count // 108000
        min_all = self.video_total_frame_count%108000 // 1800
        second_all = self.video_total_frame_count% 1800 // 30
        hours_cur = self.current_frame_no // 108000
        min_cur = self.current_frame_no%108000 // 1800
        second_cur = self.current_frame_no% 1800 // 30
        self.ui.label_framecnt.setText("{:0>2d}:{:0>2d}:{:0>2d}/{:0>2d}:{:0>2d}:{:0>2d}".format(hours_cur,min_cur,second_cur,hours_all,min_all,second_all))
        #self.ui.label_framecnt.setText(f"frame number: {self.current_frame_no}/{self.video_total_frame_count}")
        self.setslidervalue(self.current_frame_no)
        self.current_time = "{:0>2d}:{:0>2d}:{:0>2d}".format(hours_cur,min_cur,second_cur)
        return frame

    def __update_label_frame(self, frame):    
        now = time.time()
        
        if frame is not None and self.ui.checkBox_Region.isChecked():
            
            frame,self.OK = self.OE.image_detect(frame) #判定場地
            
        if self.OK is True and self.ui.checkBox_OCR.isChecked():
            if self.current_frame_no % 150 == 1:
                #self.TM.detect(frame,method = self.method)
                player,score1,score2 = self.TM.detect(frame,method = self.method)
                #print(player,score1,score2)
                self.load_info(player,score1,score2)
        
        if self.model and self.OK and self.ui.checkBox_Object.isChecked():
            x,y=0,0
            x1,y1,x2,y2 = 0,0,0,0
            players = []
            rockets = []
            img = frame.copy()
            yolo, details = yolov5_engine.predict(img,self.model)
            #print(details , "a\n")
            for item in details:
                if item[6] == "ball" :
                    x1,y1,x2,y2 = int(item[0]),int(item[1]),int(item[2]),int(item[3])
                    x,y = (x1+x2)//7.5,(y1+y2)//7.5
                if item[6] == "rocket":
                    rockets.append([int(item[0]),int(item[1]),int(item[2]),int(item[3])])
                if item[6] == "player":
                    players.append([int(item[0]),int(item[1]),int(item[2]),int(item[3])])
            # img = frame[200:800,200:800,:]
            # pose = mediapipe(img)
            # img = pose.draw_pose()
            # frame[200:800,200:800,:] = img[:,:,:]
            self.shuttle_position = np.ones((1080, 1920,3),dtype = np.uint8)
            self.shuttle_position *= 255
            if players != [] and self.ui.checkBox_Pose.isChecked():
                #t_start = time.perf_counter()
                for position in players:
                    
                    img = frame[position[1]-20:position[3]+20,position[0]-20:position[2]+20,:] 
                    if img.shape[0] == 0:
                        continue
                    # mediapipe
                    # pose = mediapipe(img)
                    # img = pose.draw_pose()
                    
                    # openpose
                    img, blank_img = detect_pose(img, self.pose_model)
                    frame[position[1] - 20:position[3] + 20, position[0] - 20:position[2] + 20, :] = img[:, :, :]
                    self.shuttle_position[position[1] - 20:position[3] + 20,
                                          position[0] - 20:position[2] + 20, :] = blank_img[:, :, :]
                    pose = self.MN.predict(blank_img)
                    cv2.putText(self.shuttle_position, pose, ((
                        position[2] + position[0]) // 2 + 20, (position[3] + position[1]) // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    
                #t_end = time.perf_counter()
                #print("MediaPipe Cost Time: ", round((t_end - t_start), 3), " sec.")
            # print(x,y)
            for rocket in rockets:
                cv2.rectangle(self.shuttle_position, (rocket[0],rocket[1]), (rocket[2],rocket[3]), (0, 255, 255), 3)
            
            if x1 and y1:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 255), 3)
                self.ball_list.put(((x1+x2)//2,(y1+y2)//2))
                if self.ball_list.qsize() > 16:
                    self.ball_list.get()
                for point in self.ball_list.queue:
                    cv2.circle(self.shuttle_position,(point[0],point[1]),3,color = (255,0,0),thickness=3)
                
                #self.shuttle_position = cv2.resize(self.shuttle_position,(288,512),interpolation=cv2.INTER_NEAREST)
            self.shuttle_position = QImage(self.shuttle_position, 1920, 1080, 3*1920, QImage.Format_RGB888).rgbSwapped()
            self.shuttle_position = QPixmap.fromImage(self.shuttle_position)
            self.shuttle_position = self.shuttle_position.scaledToHeight(288)
            self.ui.shuttle_label.setPixmap(self.shuttle_position)
            self.ui.shuttle_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            
        
        
        bytesPerline = 3 * self.video_width
        qimg = QImage(frame, self.video_width, self.video_height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(qimg)

        if self.qpixmap.width()/16 >= self.qpixmap.height()/9: # like 1600/16 > 90/9, height is shorter, align width
            self.qpixmap = self.qpixmap.scaledToWidth(self.qpixmap_fix_width)
        else: # like 1600/16 < 9000/9, width is shorter, align height
            self.qpixmap = self.qpixmap.scaledToHeight(self.qpixmap_fix_height)
        self.ui.label_videoframe.setPixmap(self.qpixmap)
        # self.ui.label_videoframe.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop) # up and left
        self.ui.label_videoframe.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter) # Center
        
        sleeptime = time.time() - now
        sleeptime = 1/self.video_fps - sleeptime
        if sleeptime > 0:
            cv2.waitKey(int(sleeptime*1000))
            
    def timer_timeout_job(self):
        if (self.videoplayer_state == "play"):
            if self.current_frame_no >= self.video_total_frame_count-1:
                #self.videoplayer_state = "pause"
                self.current_frame_no = 0 # auto replay
                self.set_current_frame_no(self.current_frame_no)
                
            else:
                self.current_frame_no += 1
                

        if (self.videoplayer_state == "stop"):
            self.current_frame_no = 0
            self.set_current_frame_no(self.current_frame_no)

        if (self.videoplayer_state == "pause"):
            self.current_frame_no = self.current_frame_no
            self.set_current_frame_no(self.current_frame_no)
            
        frame = self.__get_next_frame() 
        self.__update_label_frame(frame)
        
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())