from matplotlib.pyplot import get
from lib import find_marker
import numpy as np
import cv2
import time
import marker_dectection
import sys
import setting
import yaml
from cam_preprocessing import get_image
calibrate = False

f = open("config.yaml",'r+',encoding='utf-8')
cfg = yaml.load(f, Loader=yaml.FullLoader)

camid = cfg['camid']
data_path = cfg["data_path"]
calibration_cfg = cfg['calibration']
heightmap = cfg['heightmap']

if len(sys.argv) > 1:
    if sys.argv[1] == 'calibrate':
        calibrate = True

gelsight_version = 'Bnz'
# gelsight_version = 'HSR'

# cap = cv2.VideoCapture("data/GelSight_Twist_Test.mov")
# cap = cv2.VideoCapture("data/GelSight_Shear_Test.mov")
cap = cv2.VideoCapture(camid)
# cap = cv2.VideoCapture(1)
# upleft_x, upleft_y, downright_x, downright_y = 0,0,0,0
# def load_crop_params():
#     global upleft_x, upleft_y, downright_x, downright_y
#     crop_coordinates = np.loadtxt("crop_coordinates.txt",dtype=np.uint16)
#     [[upleft_x, upleft_y],[downright_x, downright_y]] = crop_coordinates
# Resize scale for faster image processing
setting.init()
RESCALE = setting.RESCALE

import tracking_class

m = tracking_class.tracking_class(
    N_=setting.N_, 
    M_=setting.M_, 
    fps_=setting.fps_, 
    x0_=setting.x0_, 
    y0_=setting.y0_, 
    dx_=setting.dx_, 
    dy_=setting.dy_)
# Create Mathing Class
use_cpp = True
if use_cpp:
    m = find_marker.Matching(
        N_=setting.N_, 
        M_=setting.M_, 
        fps_=setting.fps_, 
        x0_=setting.x0_, 
        y0_=setting.y0_, 
        dx_=setting.dx_, 
        dy_=setting.dy_)
"""
N_, M_: the row and column of the marker array
x0_, y0_: the coordinate of upper-left marker
dx_, dy_: the horizontal and vertical interval between adjacent markers
"""

# save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')

if gelsight_version == 'HSR':
    out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (215,215))
else:
    out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (1280//RESCALE,720//RESCALE))

# for i in range(30): ret, frame = cap.read()
abe_array = np.load('./abe_corr.npz') # change this with your aberration array 
x_index = abe_array['x']
y_index = abe_array['y']

while(True):

    # capture frame-by-frame
    ret, frame = cap.read()
    # load_crop_params()
    frame = get_image(frame)
    if not(ret):
        break

    frame_raw = frame.copy()

    # resize (or unwarp)    frame = frame[x_index, y_index, :]
    if gelsight_version == 'HSR':
        frame = marker_dectection.init_HSR(frame)
    else:
        frame = marker_dectection.init(frame)
    # frame = marker_dectection.init_HSR(frame)

    # find marker masks
    mask = marker_dectection.find_marker(frame)
    # find marker centers
    mc = marker_dectection.marker_center(mask, frame)


    if calibrate == False:
        tm = time.time()
        # # matching init
        m.init(mc)

        # # matching
        m.run()
        # print(time.time() - tm)

        # # matching result
        """
        output: (Ox, Oy, Cx, Cy, Occupied) = flow
            Ox, Oy: N*M matrix, the x and y coordinate of each marker at frame 0
            Cx, Cy: N*M matrix, the x and y coordinate of each marker at current frame
            Occupied: N*M matrix, the index of the marker at each position, -1 means inferred. 
                e.g. Occupied[i][j] = k, meaning the marker mc[k] lies in row i, column j.
        """
        flow = m.get_flow()

        # # draw flow
        marker_dectection.draw_flow(frame, flow)
        # print(np.shape(flow[0]))
        # print(flow[-1])
        # print(flow[0],np.shape(flow[0]))
    mask_img = mask.astype(frame[0].dtype)
    mask_img = cv2.merge((mask_img, mask_img, mask_img))

    # cv2.imshow('raw',frame_raw)
    cv2.imshow('frame',frame)
    # frame_hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imshow('frame in hsv', frame_hsv)
    cv2.imshow("mask",mask_img)
    if calibrate:
        # Display the mask 
        cv2.imshow('mask',mask_img)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()