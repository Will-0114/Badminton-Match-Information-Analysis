import cv2
import mediapipe as mp
import numpy as np

class mediapipe(object):
    
    def __init__(self,img):

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.image = img

    def getPosition(self,img, results, draw=False):
        lmList= []
        if results.pose_landmarks:
            h, w, c = img.shape
            for lm in results.pose_landmarks.landmark:
            # for id, lm in enumerate(results.pose_landmarks.landmark)    
            #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                # lmList.append([id, cx, cy])
                lmList.append((cx, cy))
                # if draw==True:
                #     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            return lmList

    def draw_pose(self):
        with self.mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=True) as holistic:
            
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            self.image.flags.writeable = False
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            results = holistic.process(self.image)
            #print(results.pose_landmarks)
            pos_list = self.getPosition(self.image, results, draw=False)
            
            self.image.flags.writeable = True
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            #print(pos_list)
            # Draw landmark annotation on the image.
            # mp_drawing.draw_landmarks(
            #     image,
            #     results.face_landmarks,
            #     mp_holistic.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_contours_style())
            self.mp_drawing.draw_landmarks(
                self.image,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles
                .get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            
        return self.image
            
    