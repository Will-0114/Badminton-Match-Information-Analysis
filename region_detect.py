import cv2
import numpy as np
import time


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

def main():
    img = cv2.imread("output_2-4.jpg")
    img,statue = image_detect(img)
    cv2.imshow("detect region",img)
    cv2.waitKey(0)
    
if __name__ == "__main__":
    main()