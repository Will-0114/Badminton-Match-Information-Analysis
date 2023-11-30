import torch
import os
import cv2

class yolov5_engine(object):
    
    @staticmethod
    def load_model(model_name,conf=0.5,iou=0.2):
        wd = os.getcwd()
        model = torch.hub.load(wd, 'custom', path=model_name, source='local')  # local repo
        model.conf = conf  # confidence threshold (0-1)
        model.iou = iou  # NMS IoU threshold (0-1)
        
        model.cuda()
        return model
        
    @staticmethod
    def predict(cvImg,model,resize=640):
        #wd = os.getcwd()
        # Model
        #model = torch.hub.load(wd, 'custom', path=model_name, source='local')  # local repo
        #model.conf = conf  # confidence threshold (0-1)
        #model.iou = iou  # NMS IoU threshold (0-1)
        # Image

        # Inference
        results = model(cvImg,resize)
        cvImg_out = results.render()[0] # update img
        #cv2.imshow("detection",cvImg_out)
        #cv2.waitKey(0)
        details = results.pandas().xyxy[0].values.tolist()  # img1 predictions (pandas)
        return cvImg_out,details
        
# img = cv2.imread("./Full_Court_Testing_Images/OK/Video_5_g1_870.jpg")
# model_name = 'weights/best.pt'
# #wd = os.getcwd()
# #     # Model
# model = yolov5_engine.load_model(model_name)
# cvImg,details = yolov5_engine.predict(img,model)
# cv2.imshow("predict",cvImg)
# # print(yolov5_engine.predict(img,model_name))