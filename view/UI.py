# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Main_UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        MainWindow.setMouseTracking(False)
        MainWindow.setStyleSheet("QSlider\n"
"\n"
"{\n"
"\n"
"    background-color: #ff00ff;\n"
"\n"
"border-style: outset;\n"
"\n"
"border-radius: 5px;\n"
"\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("QWidget{background-color:GRAY}")
        self.centralwidget.setObjectName("centralwidget")
        self.control_widget = QtWidgets.QTabWidget(self.centralwidget)
        self.control_widget.setGeometry(QtCore.QRect(1360, 10, 550, 960))
        self.control_widget.setIconSize(QtCore.QSize(16, 16))
        self.control_widget.setObjectName("control_widget")
        self.tab_load_file = QtWidgets.QWidget()
        self.tab_load_file.setStyleSheet("QWidget{background-color:white}")
        self.tab_load_file.setObjectName("tab_load_file")
        self.label_filepath = QtWidgets.QLabel(self.tab_load_file)
        self.label_filepath.setGeometry(QtCore.QRect(120, 10, 421, 61))
        self.label_filepath.setStyleSheet("QLabel{background-color:#BEBEBE}")
        self.label_filepath.setTextFormat(QtCore.Qt.PlainText)
        self.label_filepath.setAlignment(QtCore.Qt.AlignCenter)
        self.label_filepath.setObjectName("label_filepath")
        self.button_openfile = QtWidgets.QPushButton(self.tab_load_file)
        self.button_openfile.setGeometry(QtCore.QRect(10, 10, 101, 61))
        self.button_openfile.setStyleSheet("QPushButton{background-color:#BEBEBE}")
        self.button_openfile.setObjectName("button_openfile")
        self.shuttle_label = QtWidgets.QLabel(self.tab_load_file)
        self.shuttle_label.setGeometry(QtCore.QRect(20, 370, 512, 288))
        self.shuttle_label.setStyleSheet("QLabel{background-color:WHITE;\n"
"\n"
"border-style: outset;\n"
"\n"
"border-radius: 10px;\n"
"\n"
"border: 3px solid #000000;}")
        self.shuttle_label.setAlignment(QtCore.Qt.AlignCenter)
        self.shuttle_label.setObjectName("shuttle_label")
        self.groupBox = QtWidgets.QGroupBox(self.tab_load_file)
        self.groupBox.setGeometry(QtCore.QRect(40, 100, 261, 221))
        self.groupBox.setObjectName("groupBox")
        self.formLayoutWidget_2 = QtWidgets.QWidget(self.groupBox)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(30, 20, 211, 181))
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_2.setContentsMargins(10, 10, 1, 1)
        self.formLayout_2.setHorizontalSpacing(10)
        self.formLayout_2.setVerticalSpacing(50)
        self.formLayout_2.setObjectName("formLayout_2")
        self.radioButton_CCOEFF = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        self.radioButton_CCOEFF.setChecked(True)
        self.radioButton_CCOEFF.setObjectName("radioButton_CCOEFF")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.radioButton_CCOEFF)
        self.radioButton_NORM_CCOEFF = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        self.radioButton_NORM_CCOEFF.setObjectName("radioButton_NORM_CCOEFF")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.radioButton_NORM_CCOEFF)
        self.radioButton_CCORR = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        self.radioButton_CCORR.setObjectName("radioButton_CCORR")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.radioButton_CCORR)
        self.radioButton_SQDIFF = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        self.radioButton_SQDIFF.setChecked(False)
        self.radioButton_SQDIFF.setObjectName("radioButton_SQDIFF")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.radioButton_SQDIFF)
        self.radioButton_NORM_SQDIFF = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        self.radioButton_NORM_SQDIFF.setObjectName("radioButton_NORM_SQDIFF")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.radioButton_NORM_SQDIFF)
        self.radioButton_NORM_CCORR = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        self.radioButton_NORM_CCORR.setObjectName("radioButton_NORM_CCORR")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.radioButton_NORM_CCORR)
        self.control_widget.addTab(self.tab_load_file, "")
        self.tab_load_model = QtWidgets.QWidget()
        self.tab_load_model.setStyleSheet("QWidget{background-color:white}")
        self.tab_load_model.setObjectName("tab_load_model")
        self.button_openmodel = QtWidgets.QPushButton(self.tab_load_model)
        self.button_openmodel.setGeometry(QtCore.QRect(10, 10, 101, 61))
        self.button_openmodel.setStyleSheet("QPushButton{background-color:#BEBEBE}")
        self.button_openmodel.setObjectName("button_openmodel")
        self.label_model_name = QtWidgets.QLabel(self.tab_load_model)
        self.label_model_name.setGeometry(QtCore.QRect(120, 10, 421, 61))
        self.label_model_name.setStyleSheet("QLabel{background-color:#BEBEBE}")
        self.label_model_name.setTextFormat(QtCore.Qt.PlainText)
        self.label_model_name.setAlignment(QtCore.Qt.AlignCenter)
        self.label_model_name.setObjectName("label_model_name")
        self.SpinBox_Conf = QtWidgets.QDoubleSpinBox(self.tab_load_model)
        self.SpinBox_Conf.setGeometry(QtCore.QRect(100, 120, 101, 31))
        self.SpinBox_Conf.setMaximum(0.99)
        self.SpinBox_Conf.setSingleStep(0.01)
        self.SpinBox_Conf.setProperty("value", 0.5)
        self.SpinBox_Conf.setObjectName("SpinBox_Conf")
        self.label = QtWidgets.QLabel(self.tab_load_model)
        self.label.setGeometry(QtCore.QRect(30, 120, 51, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.tab_load_model)
        self.label_2.setGeometry(QtCore.QRect(30, 170, 51, 31))
        self.label_2.setObjectName("label_2")
        self.SpinBox_Iou = QtWidgets.QDoubleSpinBox(self.tab_load_model)
        self.SpinBox_Iou.setGeometry(QtCore.QRect(100, 170, 101, 31))
        self.SpinBox_Iou.setMaximum(0.99)
        self.SpinBox_Iou.setSingleStep(0.01)
        self.SpinBox_Iou.setProperty("value", 0.2)
        self.SpinBox_Iou.setObjectName("SpinBox_Iou")
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab_load_model)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 510, 251, 241))
        self.groupBox_2.setObjectName("groupBox_2")
        self.checkBox_Region = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_Region.setGeometry(QtCore.QRect(30, 40, 181, 16))
        self.checkBox_Region.setObjectName("checkBox_Region")
        self.checkBox_OCR = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_OCR.setGeometry(QtCore.QRect(30, 90, 101, 16))
        self.checkBox_OCR.setObjectName("checkBox_OCR")
        self.checkBox_Object = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_Object.setGeometry(QtCore.QRect(30, 140, 121, 16))
        self.checkBox_Object.setObjectName("checkBox_Object")
        self.checkBox_Pose = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_Pose.setGeometry(QtCore.QRect(30, 190, 131, 16))
        self.checkBox_Pose.setObjectName("checkBox_Pose")
        self.button_openposemodel = QtWidgets.QPushButton(self.tab_load_model)
        self.button_openposemodel.setGeometry(QtCore.QRect(10, 230, 101, 61))
        self.button_openposemodel.setStyleSheet("QPushButton{background-color:#BEBEBE}")
        self.button_openposemodel.setObjectName("button_openposemodel")
        self.label_pose_model_name = QtWidgets.QLabel(self.tab_load_model)
        self.label_pose_model_name.setGeometry(QtCore.QRect(120, 230, 421, 61))
        self.label_pose_model_name.setStyleSheet("QLabel{background-color:#BEBEBE}")
        self.label_pose_model_name.setTextFormat(QtCore.Qt.PlainText)
        self.label_pose_model_name.setAlignment(QtCore.Qt.AlignCenter)
        self.label_pose_model_name.setObjectName("label_pose_model_name")
        self.button_mobilenetmodel = QtWidgets.QPushButton(self.tab_load_model)
        self.button_mobilenetmodel.setGeometry(QtCore.QRect(10, 330, 101, 61))
        self.button_mobilenetmodel.setStyleSheet("QPushButton{background-color:#BEBEBE}")
        self.button_mobilenetmodel.setObjectName("button_mobilenetmodel")
        self.label_mobilenet_model_name = QtWidgets.QLabel(self.tab_load_model)
        self.label_mobilenet_model_name.setGeometry(QtCore.QRect(120, 330, 421, 61))
        self.label_mobilenet_model_name.setStyleSheet("QLabel{background-color:#BEBEBE}")
        self.label_mobilenet_model_name.setTextFormat(QtCore.Qt.PlainText)
        self.label_mobilenet_model_name.setAlignment(QtCore.Qt.AlignCenter)
        self.label_mobilenet_model_name.setObjectName("label_mobilenet_model_name")
        self.control_widget.addTab(self.tab_load_model, "")
        self.vision_widget = QtWidgets.QTabWidget(self.centralwidget)
        self.vision_widget.setGeometry(QtCore.QRect(10, 20, 1330, 821))
        self.vision_widget.setObjectName("vision_widget")
        self.Main_player = QtWidgets.QWidget()
        self.Main_player.setStyleSheet("QWidget{background-color:white}")
        self.Main_player.setObjectName("Main_player")
        self.label_videoframe = QtWidgets.QLabel(self.Main_player)
        self.label_videoframe.setGeometry(QtCore.QRect(20, 10, 1280, 720))
        self.label_videoframe.setStyleSheet("QLabel{background-color:WHITE;\n"
"\n"
"border-style: outset;\n"
"\n"
"border-radius: 10px;\n"
"\n"
"border: 3px solid #000000;}")
        self.label_videoframe.setScaledContents(False)
        self.label_videoframe.setAlignment(QtCore.Qt.AlignCenter)
        self.label_videoframe.setObjectName("label_videoframe")
        self.slider_videoframe = QtWidgets.QSlider(self.Main_player)
        self.slider_videoframe.setGeometry(QtCore.QRect(150, 750, 1031, 22))
        self.slider_videoframe.setStyleSheet("QSlider\n"
"\n"
"{\n"
"\n"
"    background-color: #000000;\n"
"\n"
"border-style: outset;\n"
"\n"
"border-radius: 10px;\n"
"\n"
"}\n"
"QSlider::groove:horizontal\n"
"\n"
"{\n"
"height: 12px;\n"
"\n"
"background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f1f1f1, stop:1 #b1b1b1);\n"
"\n"
"margin: 3px 0\n"
"\n"
"}\n"
"QSlider::handle:horizontal\n"
"\n"
"{\n"
"\n"
"background: QRadialGradient(cx:0, cy:0, radius: 1, fx:0.5, fy:0.5, stop:0 white, stop:1 black);\n"
"\n"
"width: 16px;\n"
"\n"
"height: 16px;\n"
"\n"
"margin: -5px 6px -5px 6px;\n"
"\n"
"border-radius:11px;\n"
"\n"
"border: 3px solid #c1ffff;\n"
"\n"
"}\n"
"\n"
"")
        self.slider_videoframe.setOrientation(QtCore.Qt.Horizontal)
        self.slider_videoframe.setObjectName("slider_videoframe")
        self.label_framecnt = QtWidgets.QLabel(self.Main_player)
        self.label_framecnt.setGeometry(QtCore.QRect(1190, 750, 111, 21))
        self.label_framecnt.setStyleSheet("")
        self.label_framecnt.setObjectName("label_framecnt")
        self.button_play = QtWidgets.QPushButton(self.Main_player)
        self.button_play.setGeometry(QtCore.QRect(90, 739, 51, 51))
        self.button_play.setStyleSheet("QPushButton{background-color:rgba(255, 255, 127, 0)}")
        self.button_play.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../icon/play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_play.setIcon(icon)
        self.button_play.setIconSize(QtCore.QSize(40, 40))
        self.button_play.setObjectName("button_play")
        self.button_stop = QtWidgets.QPushButton(self.Main_player)
        self.button_stop.setGeometry(QtCore.QRect(30, 740, 50, 50))
        self.button_stop.setStyleSheet("QPushButton{background-color:rgba(255, 255, 127, 0)}")
        self.button_stop.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../icon/restart.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_stop.setIcon(icon1)
        self.button_stop.setIconSize(QtCore.QSize(40, 40))
        self.button_stop.setObjectName("button_stop")
        self.vision_widget.addTab(self.Main_player, "")
        self.RGB_player = QtWidgets.QWidget()
        self.RGB_player.setStyleSheet("QWidget{background-color:white}")
        self.RGB_player.setObjectName("RGB_player")
        self.tableWidget = QtWidgets.QTableWidget(self.RGB_player)
        self.tableWidget.setGeometry(QtCore.QRect(10, 10, 1291, 771))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(6, item)
        self.vision_widget.addTab(self.RGB_player, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.control_widget.setCurrentIndex(1)
        self.vision_widget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.button_stop, self.button_play)
        MainWindow.setTabOrder(self.button_play, self.slider_videoframe)
        MainWindow.setTabOrder(self.slider_videoframe, self.vision_widget)
        MainWindow.setTabOrder(self.vision_widget, self.button_openfile)
        MainWindow.setTabOrder(self.button_openfile, self.control_widget)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.control_widget.setToolTip(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.label_filepath.setText(_translate("MainWindow", "File Name"))
        self.button_openfile.setText(_translate("MainWindow", "Openfile"))
        self.shuttle_label.setText(_translate("MainWindow", "Shuttle"))
        self.groupBox.setTitle(_translate("MainWindow", "Region Datect Model"))
        self.radioButton_CCOEFF.setText(_translate("MainWindow", "CCOEFF"))
        self.radioButton_NORM_CCOEFF.setText(_translate("MainWindow", "NORM_CCOEFF"))
        self.radioButton_CCORR.setText(_translate("MainWindow", "CCORR"))
        self.radioButton_SQDIFF.setText(_translate("MainWindow", "SQDIFF"))
        self.radioButton_NORM_SQDIFF.setText(_translate("MainWindow", "NORM_SQDIFF"))
        self.radioButton_NORM_CCORR.setText(_translate("MainWindow", "NORM_CCORR"))
        self.control_widget.setTabText(self.control_widget.indexOf(self.tab_load_file), _translate("MainWindow", "Tab 1"))
        self.button_openmodel.setText(_translate("MainWindow", "LoadModel"))
        self.label_model_name.setText(_translate("MainWindow", "yolov5 Model Name"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Conf</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">Iou</span></p></body></html>"))
        self.groupBox_2.setTitle(_translate("MainWindow", "GroupBox"))
        self.checkBox_Region.setText(_translate("MainWindow", "Region Detect"))
        self.checkBox_OCR.setText(_translate("MainWindow", "OCR Detext"))
        self.checkBox_Object.setText(_translate("MainWindow", "Object Detect"))
        self.checkBox_Pose.setText(_translate("MainWindow", "Pose Detect"))
        self.button_openposemodel.setText(_translate("MainWindow", "LoadModel"))
        self.label_pose_model_name.setText(_translate("MainWindow", "Openpose Model Name"))
        self.button_mobilenetmodel.setText(_translate("MainWindow", "LoadModel"))
        self.label_mobilenet_model_name.setText(_translate("MainWindow", "Mobilenet Model Name"))
        self.control_widget.setTabText(self.control_widget.indexOf(self.tab_load_model), _translate("MainWindow", "Tab 2"))
        self.label_videoframe.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:48pt; font-weight:600;\">Video Player</span></p></body></html>"))
        self.label_framecnt.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#000000;\">00:00:00/00:00:00</span></p></body></html>"))
        self.vision_widget.setTabText(self.vision_widget.indexOf(self.Main_player), _translate("MainWindow", "Main"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Time"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Player1"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Player2"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "First"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Second"))
        item = self.tableWidget.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "Third"))
        item = self.tableWidget.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow", "Match_Score"))
        self.vision_widget.setTabText(self.vision_widget.indexOf(self.RGB_player), _translate("MainWindow", "Competition Information"))
