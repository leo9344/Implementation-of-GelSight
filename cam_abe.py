import cv2
import numpy as np
import glob

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob("./data/sample_*.jpg")
i=0;
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
    #print(corners)

    if ret:

        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        #print(corners2)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (8, 6), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        i+=1;
        cv2.imwrite('conimg'+str(i)+'.jpg', img)
        cv2.waitKey(1500)

print(len(img_points))
cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx) # 内参数矩阵
print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
print("tvecs:\n", tvecs ) # 平移向量  # 外参数

print("-----------------------------------------------------")

img = cv2.imread(images[2])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#显示更大范围的图片（正常重映射之后会删掉一部分图像）
print (newcameramtx)
print("------------------使用undistort函数-------------------")
dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
x,y,w,h = roi
dst1 = dst[y:y+h,x:x+w]
cv2.imwrite('calibresult3.jpg', dst1)
np.savetxt("./mtx.txt",mtx)
np.savetxt("./dist.txt",dist)
print ("方法一:dst的大小为:", dst1.shape)

camid: 0

sample:
  from: 1
  to: 30

data_path: ./data

crop: True

marker:
  x0_: 10 #/ RESCALE
  y0_: 17 #/ RESCALE
  dx_: 40 #/ RESCALE
  dy_: 40 #/ RESCALE
  N_: 5
  M_: 5

calibration:
  upleft_x: 170
  upleft_y: 83
  downright_x: 434
  downright_y: 318
  method: camera # marker or camera
  dist: [1.429079069840946374e+02,-5.049435358457427355e+04,3.146517744641259551e-01,
        -1.511966389672928131e-02,-2.325254239025613856e+02]
  mtx: [[5.114054972960955638e+03,0.000000000000000000e+00,3.033560857139311793e+02],
        [0.000000000000000000e+00,6.345695124741401742e+03,2.373591659885894387e+02],
        [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]]
  BallRad: 3 #4.76/2 #mm
  Pixmm: 0.031067    #.10577 #4.76/100 #0.0806 * 1.5 mm/pixel # 0.03298 = 3.40 / 103.0776
  ratio: 0.5
  red_range: [-80, 90]  # [-45, 45]
  green_range: [-90, 90] #[-60, 50]
  blue_range: [-90, 90] # [-80, 60]
  zeropoint: [-90, -90, -90] 
  lookscale: [180., 180., 180.]

heightmap:
  cpp_enable: False
  abe_path: ./abe_corr.npz
  table_path: ./table.npy
  table_account_path: ./count_map.py
  table_smooth_path: ./table_smooth.npy

tracking:
  lib_path: ./lib
  RESCALE: 1
  fps_: 30
