import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml

f = open("config.yaml",'r+',encoding='utf-8')
cfg = yaml.load(f, Loader=yaml.FullLoader)

camid = cfg['camid']
data_path = cfg["data_path"]
calibration = cfg['calibration']
method = calibration['method']
crop = cfg['crop']

downright_x = calibration['downright_x']
downright_y = calibration['downright_y']
upleft_x = calibration['upleft_x']
upleft_y = calibration['upleft_y']

abe_path = cfg['heightmap']['abe_path']
abe_array = np.load(abe_path)
x_index = abe_array['x']
y_index = abe_array['y']

def get_image(img):
    if method == 'camera':
        mtx = np.array(calibration['mtx'])
        dist = np.array(calibration['dist'])
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
        x,y,w,h = roi
        dst1 = dst[y:y+h,x:x+w]
        ref_img = dst1[upleft_y:downright_y+1, upleft_x:downright_x+1]
        return ref_img
    
    elif method == 'marker':
        ref_img = img[upleft_y:downright_y+1, upleft_x:downright_x+1]
        ref_img = ref_img[x_index, y_index, :]
        return ref_img
    
# img = cv2.imread("./data/ref.jpg")
# cv2.imshow("res",get_image(img))
# cv2.waitKey(0)