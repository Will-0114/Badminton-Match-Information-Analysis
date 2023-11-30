import cv2
import numpy as np
import time
from paddleocr import PaddleOCR

class opencv_engine(object):
    
    @staticmethod
    def point_float_to_int(point):
        return (int(point[0]), int(point[1]))

    @staticmethod
    def read_image(file_path):
        return cv2.imread(file_path)

    @staticmethod
    def draw_point(img, point=(0, 0), color = (0, 0, 255)): # red
        point = opencv_engine.point_float_to_int(point)
        print(f"get {point}")
        point_size = 1
        thickness = 4
        return cv2.circle(img, point, point_size, color, thickness)
    
    @staticmethod
    def getvideoinfo(video_path): 
        # https://docs.opencv.org/4.5.3/dc/d3d/videoio_8hpp.html
        videoinfo = {}
        vc = cv2.VideoCapture(video_path)
        videoinfo["vc"] = vc
        videoinfo["fps"] = vc.get(cv2.CAP_PROP_FPS)
        videoinfo["frame_count"] = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        videoinfo["width"] = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        videoinfo["height"] = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return videoinfo
    
    @staticmethod
    def image_detect(cvImg):
        cvImg_h, cvImg_w, cvImg_c = cvImg.shape
        Img_hsv = cv2.cvtColor(cvImg, cv2.COLOR_BGR2HSV)

        ## set the range to trace: green
        lower_color_bounds_green = np.array([12, 52, 69]) #cv2.Scalar(100, 0, 0)  ##
        upper_color_bounds_green =  np.array([101,193,255])  #cv2.Scalar(225,80,80) ##
        mask_green = cv2.inRange(Img_hsv, lower_color_bounds_green, upper_color_bounds_green)
        mask_green_rgb = cv2.cvtColor(mask_green,cv2.COLOR_GRAY2BGR)
        cvImg_green = cvImg & mask_green_rgb

        ##判別_green
        img_green_gray = cv2.cvtColor(cvImg_green, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(img_green_gray, 180, 255,cv2.THRESH_BINARY_INV)
        
        kernel = np.ones((6, 3), np.uint8)
        erode_green = cv2.erode(img_green_gray, kernel, iterations=5)
        kernel = np.ones((5, 5), np.uint8)
        dilate_green = cv2.dilate(erode_green, kernel, iterations = 10)
        
        contours_green, hier = cv2.findContours(dilate_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bbox = 0
        ##增加限制式
        for c in contours_green:

            # 輸出：(x, y)矩形左上角座標、w 矩形寬(x軸方向)、h 矩形高(y軸方向)
            (x, y, w, h) = cv2.boundingRect(c)
            
            ##限制式
            if w*h < 0.3 * cvImg_w * cvImg_h:
                pass
            else:
                #找上面兩頂點
                min_y = 1000
                min_y_list = []
                for point in c:
                    # print(f"point:{point}")
                    if point[0][1] < min_y:
                        min_y = point[0][1]
                    
                for point in c:
                    if point[0][1] < min_y + 30:
                        min_y_list.append(point[0])
                
                min_x = 1000
                max_x = 0
                for point in min_y_list:
                    if point[0] < min_x:
                        min_x = point[0]
                    if point[0] > max_x:
                        max_x = point[0]
                
                point_upper_left = (min_x, min_y)
                point_upper_right = (max_x, min_y)
                # cvImg = cv2.circle(cvImg, (min_x,min_y), radius=0, color=(0, 0, 255), thickness=10)
                # cvImg = cv2.circle(cvImg, (max_x,min_y), radius=0, color=(0, 0, 255), thickness=10)

                #找下面兩頂點
                max_y = 0
                max_y_list = []
                for point in c:
                    # print(f"point:{point}")
                    if point[0][1] > max_y:
                        max_y = point[0][1]
                    
                for point in c:
                    if point[0][1] > max_y - 30:
                        max_y_list.append(point[0])
                
                min_x = 1000
                max_x = 0
                for point in max_y_list:
                    if point[0] < min_x:
                        min_x = point[0]
                    if point[0] > max_x:
                        max_x = point[0]
                
                point_lower_left = (min_x, max_y)
                point_lower_right = (max_x, max_y)
                point_lower_middle = ((min_x + max_x)/2, max_y)
                slope = (point_upper_right[1]-point_lower_right[1])/(point_upper_right[0]-point_lower_right[0])

                #依頂點判別
                if point_upper_left[0] < 200 or cvImg_w - point_upper_right[0] <200:
                    pass
                elif abs(cvImg_w - point_upper_right[0] - point_upper_left[0]) > 200:
                    pass
                elif point_lower_left[0] < 30 or cvImg_w - point_lower_right[0] <30:
                    pass
                #用斜率進一步篩選
                elif slope > 3:
                    pass 
                else:                     
                    bbox+=1


        if bbox != 0:
            text = 'OK'
            status = True
            #num_OK +=1
            #out_folder = folder + "2OK"
            # print("OK")
            cv2.putText(cvImg, text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX,5, (0, 255, 0), 3, cv2.LINE_AA)
            # cv2.imwrite(f"./output/{out_folder}/{fn_name}", cvImg)
        else:
            text = 'NG'
            status = False
            #num_NG +=1
            #out_folder = folder + "2NG"
            cv2.putText(cvImg, text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX,5, (0, 0, 255), 3, cv2.LINE_AA)
            # cv2.imwrite(f"./output/{out_folder}/{fn_name}", cvImg)

        return cvImg,status
    
class template_matching(object):
    
    def __init__(self):
        self.ocr_model = PaddleOCR(lang='en', use_gpu=True,
                        cls_model_dir="ocrtest/paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer",
                        det_model_dir="ocrtest/paddleocr/whl/det/en/en_PP-OCRv3_det_infer",
                        rec_model_dir="ocrtest/paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer")
    
    @staticmethod
    def removeSame(pts, threshold):  # Threshold = distance
        elements = []
        for x, y in pts:
            for ele in elements:
                if ((x-ele[0])**2 + (y-ele[1])**2) < threshold**2:
                    break
            else:
                elements.append((x, y))

        return elements

    #@staticmethod
    def template_match(self,img_color, template,threshold = 0.8, method=2, isShow=True):
        t_start = time.perf_counter()
        if len(img_color.shape) == 3:
            img_c = img_color.copy()
            img_gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR)
            img_c = img_color.copy()

        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()

        w, h = template.shape[::-1]
        # All the 6 methods for comparison in a list
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        # if not method in methods:
        #     print("Not valid method")
        #     return
        #print("Matching method", methods[method])
        res = cv2.matchTemplate(img_gray, template_gray, method)
        # print(res)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        t_end = time.perf_counter()
        print("Matching Time: ", round((t_end - t_start), 3), " sec.")
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        score = 0
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:  # min is matched
            top_left = min_loc
            score = (1-min_val)
        else:
            top_left = max_loc
            score = max_val
        bottom_right = (top_left[0] + w+400, top_left[1] + h)
        #print(score)
        # if score > threshold:
        #     cv2.rectangle(img_c, top_left, bottom_right, (255, 255, 0), 5)
        if isShow:
            cv2.imshow("Matching Score Matrix:", cv2.normalize(
                res, None, 0, 1, cv2.NORM_MINMAX))
            cv2.imshow("Match Result:", img_c)
            cv2.waitKey(0)
        return img_c, res, score ,top_left , bottom_right
    
    #@staticmethod
    def template_match_multiple(img_color, template, method=0, threshold=0.8, isShow=True):
        t_start = time.perf_counter()
        img_c = img_color.copy()
        img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
        w, h = template.shape[::-1]
        # All the 6 methods for comparison in a list
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        print("Matching method", methods[method])
        #m = eval(methods[method])
        res = cv2.matchTemplate(img, template, method)
        res = cv2.normalize(res, None, 0, 1, cv2.NORM_MINMAX)  # 正規化

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:  # min is matched
            res = 1 - res
            #print(res)
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        t_end = time.perf_counter()
        print("Matching Time: ", round((t_end - t_start), 3), " sec.")
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        loc = np.where(res >= threshold)  # loc: [[x1, x2, ...] [y1, y2, ....]]
        #print(len(loc[0]))  # none max suppresion
        for pt in zip(*loc[::-1]):  # 配對位置 (x, y)
            cv2.rectangle(img_c, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
        if isShow:
            cv2.imshow(methods[method] + "Matching Score Matrix:", res)
            cv2.imshow(methods[method] + "Match Result:", img_c)
            cv2.waitKey(0)
        return img_c, res

    #@staticmethod
    def draw_cross(image, pt, length=5, color=(0, 0, 255), lineWidth=1, isShow=True):
        (x, y) = pt
        start_pt = (x-length, y)
        end_pt = (x+length, y)
        image = cv2.line(image, start_pt, end_pt, color, lineWidth)
        start_pt = (x, y-length)
        end_pt = (x, y + length)
        image = cv2.line(image, start_pt, end_pt, color, lineWidth)
        if isShow:
            cv2.imshow("draw cross image", image)
            cv2.waitKey(0)
        return image

    #@staticmethod
    def find_contour(img, threshold=127, isBlur=True, isShow=True, color=(0, 0, 255)):
        if len(img.shape) == 3:
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            imgray = img.copy()
        if isBlur:
            imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
        # remember: object is white in a black background
        ret, thresh = cv2.threshold(imgray, threshold, 255, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # [(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 0, 0), (255, 255, 0),(0, 0, 255), (0, 255, 255), (255, 0, 255),]
        color = color
        img_contour = img.copy()
        for i in range(len(contours)):
            cnt = contours[i]
            cv2.drawContours(img_contour, [cnt], 0, color, 1)
        # print(contours[0])
        # print(hierarchy)
        if isShow:
            cv2.imshow("Gray image", imgray)
            cv2.imshow("Threshold", thresh)
            cv2.imshow("contours", img_contour)
            cv2.imshow("image", img)
            cv2.waitKey(0)
        return contours, img_contour

    #@staticmethod
    def calc_contour_feature(img, contours, isShow=True):
        """
        輸入 contours
        回傳: feature list
        """
        feature_list = list()
        for cont in contours:
            area = cv2.contourArea(cont)
            if area == 0:
                continue
            perimeter = cv2.arcLength(cont, closed=True)
            bbox = cv2.boundingRect(cont)
            # print(bbox)
            bbox2 = cv2.minAreaRect(cont)
            # print(bbox2)
            circle = cv2.minEnclosingCircle(cont)
            if len(cont) > 5:
                ellipes = cv2.fitEllipse(cont)
            else:
                ellipes = None
            # print(ellipes)
            # Moment
            M = cv2.moments(cont)  # return all moment of given contour
            if area != 0:  # same as M["m00"] !=0
                cx = int(M["m10"] / M["m00"]+0.5)
                cy = int(M["m01"] / M["m00"]+0.5)
                center = (cx, cy)
            else:
                center = (None, None)
            feature = (center, area, perimeter, bbox, bbox2, circle, ellipes)
            feature_list.append(feature)
        return feature_list

    #@staticmethod
    def crop_image(image, bbox):
        image_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return image_crop

    #@staticmethod
    def template_match_scale(img_color, template_color, method=0, threshold=0.9, minScale=0.5, maxScale=1.5,  isShow=True):
        """
        threshold: 相似度
        minScale, maxScale: 將影像放大縮小的範圍
        """
        t_start = time.perf_counter()
        img_c = img_color.copy()
        if len(img_color.shape) == 3:
            img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
        else:
            img = img_color.copy()

        if len(template_color.shape) == 3:
            template = cv2.cvtColor(template_color.copy(), cv2.COLOR_BGR2GRAY)
        else:
            template = template_color.copy()
        w, h = template.shape[::-1]
        # All the 6 methods for comparison in a list
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        #print("Matching method", methods[method])
        m = eval(methods[method])
        max = 0  # max matching score
        max_scale = 0
        max_index = 0  # record the image index
        i = 0
        #res_list = list()
        max_res = None
        # find the most similar scale
        for scale in np.arange(minScale, maxScale, 0.1):
            temp = cv2.resize(template.copy(), None, fx=scale, fy=scale)
            # print(temp.shape)
            res = cv2.matchTemplate(img, temp, method)

            if m == 5:  # min is matched
                res = 1 - res
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > max:
                max = max_val
                max_index = i  # image index
                max_scale = scale
                max_res = res.copy()
                max_res_loc = max_loc
            i = i + 1
            # res_list.append(res)
        #print("Found the most similar scale: ", max_scale, max_index)
        # find the location in max_index image
        map = np.where(res > threshold, 255, 0)
        loc = np.where(map >= threshold)
        # print(loc)
        map = np.array(map, dtype=np.uint8)
        map = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
        # map_thinned = cv2.ximgproc.thinning(map)  ## cannot find the center of the blob
        contours, img_contour = template_matching.find_contour(
            map, threshold=127, isBlur=False, isShow=False, color=(0, 0, 255))
        feature_list = template_matching.calc_contour_feature(img, contours, isShow=False)
        # Note find centroid is not good, so try to find the max
        pt_list = list()
        for f in feature_list:
            box = f[3]
            box = [box[0], box[1], box[0]+box[2], box[1] + box[3]]
            res_roi = template_matching.crop_image(max_res, box)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_roi)
            pt_list.append((max_loc[0] + box[0], max_loc[1]+box[1]))
        # print(pt_list)
        h = int(h * max_scale)
        w = int(w * max_scale)
        #print(h," : ", w)
        t_end = time.perf_counter()
        print("Matching Time: ", round((t_end - t_start), 3), " sec.")
        max_res = cv2.cvtColor(np.array(max_res*255, np.uint8), cv2.COLOR_GRAY2BGR)
        max_res_disp = max_res.copy()
        for pt in pt_list:  # zip(*loc[::-1]): ## 配對位置 (x, y)
            cv2.rectangle(
                img_c, pt, (int(pt[0]) + w, int(pt[1]) + h), (0, 0, 255), 5)
            max_res_disp = template_matching.draw_cross(max_res_disp, pt, isShow=False)
        if isShow:
            cv2.imshow(methods[m] + "Matching Score Matrix:", max_res_disp)
            cv2.imshow(methods[m] + "Match Result:", img_c)
            cv2.waitKey(0)
        return img_c, max_res

    def ocr_name(self,img: np.ndarray) -> list:
        cut_img = 255-img
        result = self.ocr_model.ocr(img, cls=False)
        txts = [line[1][0] for line in result]
        #print(txts)
        
        if not txts:
            return None,None
        return txts

    def ocr_score(self,img: np.ndarray) -> list:

        cut_img = 255-img
        result = self.ocr_model.ocr(img, cls=False)
        txts = [line[1][0] for line in result]
        #print(txts)
        
        if not txts:
            return None,None
        
        
        p1,p2 = txts[:len(txts)//2],txts[len(txts)//2:]
        return p1,p2

    
        
    
    def detect(self,Img,method,threshold = 0.99):
        #img_rgb = cv2.imread('./images/Switch2.bmp')
        #img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('./images/template/bwf_logo_template.bmp', 0)
        #threshold = 0.99
        Img, res, score ,top_left , bottom_right= self.template_match(Img, template, method =method, threshold = threshold, isShow = False)
        
        Img = Img[top_left[1]-10:bottom_right[1]+10,top_left[0]:bottom_right[0]-20]
        # cv2.imshow("img",Img)
        # cv2.waitKey(0)
        if isinstance(Img,np.ndarray):
            left_Img = Img[:,:Img.shape[1]//2]
            right_Img = Img[:,Img.shape[1]//2:]
            player = self.ocr_name(left_Img)
            score1,score2 = self.ocr_score(right_Img)
            if (score1 == None) or (score2 == None):
                return None,None,None
            score1 += ["-"]*(3-len(score1))
            score2 += ["-"]*(3-len(score2))
            #p1,p2 = self.ocr_score(left_Img)
            
            return player,score1,score2