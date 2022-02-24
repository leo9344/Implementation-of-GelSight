import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
import yaml
import argparse
import glob

class imp:
    def __init__(self):
        self.kernel = self.make_kernal(5, 'circle')
        self.marker_dis_thre = 42
        self.position_list = []

        self.img_copy = []
        self.img = []

        self.points = []
        self.count = 0
        self.upleft_x = 170
        self.upleft_y = 83
        self.downright_x = 434
        self.downright_y = 318


    def make_kernal(self, n, type):
        if type == 'circle':
            kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
        else:
            kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))
        return kernal

    def defect_mask(self, pad):
        y_d = self.downright_y+1 - self.upleft_y
        x_d = self.downright_x+1 - self.upleft_x
        mask = np.ones((y_d, x_d)) #320, 427
        mask[:pad, :] = 0
        mask[-pad:, :] = 0
        mask[:, :pad] = 0
        mask[:, -pad:] = 0
        return mask

    def mask_marker(self, raw_image):
        m, n = raw_image.shape[1], raw_image.shape[0]
        raw_image = cv2.pyrDown(raw_image).astype(np.float32)
        blur = cv2.GaussianBlur(raw_image, (25, 25), 0)
        blur2 = cv2.GaussianBlur(raw_image, (5, 5), 0)
        diff = blur - blur2
        diff *= 16.0

        # cv2.imshow('blur2', blur.astype(np.uint8))
        # cv2.waitKey(1)

        diff[diff < 0.] = 0.
        diff[diff > 255.] = 255.

        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        # cv2.imshow('diff', diff.astype(np.uint8))
        # cv2.waitKey(1)

        mask_b = diff[:, :, 0] > 150    #150
        mask_g = diff[:, :, 1] > 150    #150
        mask_r = diff[:, :, 2] > 150    #150
        mask = (mask_b * mask_g + mask_b * mask_r + mask_g * mask_r) > 0

        # cv2.imshow('mask', mask.astype(np.uint8) * 255)
        cv2.waitKey(0)
        # cv2.waitKey(1)
        mask = cv2.resize(mask.astype(np.uint8), (m, n))
        mask = cv2.dilate(mask, self.kernel, iterations=1) * self.dmask
        # cv2.imshow('mask', mask.astype(np.uint8) * 255)
        # mask = cv2.erode(mask, self.kernal4, iterations=1)
        return (1 - mask) * 255

    def find_dots(self, binary_image):
        # down_image = cv2.resize(binary_image, None, fx=2, fy=2)
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 1
        params.maxThreshold = 12
        params.minDistBetweenBlobs = 9
        params.filterByArea = True
        params.minArea = 15
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minInertiaRatio = 0.5
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary_image.astype(np.uint8))
        # im_to_show = (np.stack((binary_image,)*3, axis=-1)-100)
        return keypoints

    def onMouse(self,event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.posList.append((x, y))

    def get_sortedarray(self, im, keypoints, display=False):
        x, y, xy = [], [], []
        print(f"keypoint size is {len(keypoints)}")
        for i in range(len(keypoints)):
            x.append(keypoints[i].pt[0])
            y.append(keypoints[i].pt[1])
            xy.append((keypoints[i].pt[1], keypoints[i].pt[0]))
        xy = sorted(xy)
        temp = []
        xy_array = []
        for i in range(len(xy)):
            y_temp, x_temp = xy[i]

            if temp:
                sum_y = 0
                for x, y in temp:
                    sum_y += y

                temp_array = np.array(temp)
                diff = np.min(np.abs(x_temp - temp_array[:, 0]))
                # print(f"{i} th diff is {diff}")
                factor1 = abs(sum_y / len(temp) - y_temp)
                # print(f"{i} th factor1 is {factor1}")

                if factor1 < self.marker_dis_thre and diff > 10:
                    temp.append((x_temp, y_temp))
                    # print(f"This is the {i} th loop")
                else:
                    mask_temp = np.zeros_like(im[:, :, 0])
                    for x, y in temp:
                        cv2.ellipse(mask_temp, (int(x), int(y)), (1, 1), 0, 0, 360, (255), -1)
                    cv2.imshow('img_test', mask_temp)
                    cv2.waitKey(0)
                    # number = int(input(f"Time {i} Enter the number of misclassified point: "))
                    number = 0
                    temp_new = []
                    while number > 0:
                        temp_new.append(temp.pop())
                        number -= 1
                    if len(temp) > 3:
                        temp = sorted(temp)
                        xy_array.append(temp)
                    temp = []
                    temp_new.reverse()
                    temp += temp_new
                    temp.append((x_temp, y_temp))
            else:
                temp.append((x_temp, y_temp))

        xy_array.append(sorted(temp))

        if display:
            for i in range(len(xy_array)):
                mask_temp = np.zeros_like(im[:, :, 0])
                for j in range(len(xy_array[i])):
                    x, y = xy_array[i][j]
                    cv2.ellipse(mask_temp, (int(x), int(y)), (1, 1), 0, 0, 360, (255), -1)
                cv2.imshow('img_test', mask_temp)
                cv2.waitKey(0)

        print(f"temp size is {np.array(xy_array).shape}")
        # print(f"xy array is {np.array(xy_array)}")
        return xy_array

    def get_corrarray(self, init_array):
        corr_array = []
        for col in init_array:
            y1, x1 = col[0]
            y2, x2 = col[-1]
            num = len(col)
            temp = [(y1, x1)]
            for i in range(1, num - 1):
                x_new = x1 + (x2 - x1) / (num - 1) * i
                y_new = y1 + (y2 - y1) / (num - 1) * i
                temp.append((y_new, x_new))
            temp.append((y2, x2))
            corr_array.append(temp)

        stand_row = np.array(corr_array[int(len(corr_array) / 2)])

        for i in range(len(corr_array)):
            for j in range(len(corr_array[i])):
                x_, y_ = corr_array[i][j]
                diff = np.abs(x_ - stand_row[:, 0])
                if np.min(diff) < 30:
                    index = np.argmin(diff)
                    corr_array[i][j] = (stand_row[index][0], y_)
        return corr_array

    def convert_format(self, array):
        array_ = []
        for item in array:
            array_ += item[:]
        return np.array(array_)

    def interp(self, corr_array, init_array, x_mesh, y_mesh):
        rbfi_x = Rbf(corr_array[:, 0], corr_array[:, 1], init_array[:, 0], function='cubic')
        rbfi_y = Rbf(corr_array[:, 0], corr_array[:, 1], init_array[:, 1], function='cubic')

        x_index = rbfi_x(x_mesh, y_mesh).astype(int)
        y_index = rbfi_y(x_mesh, y_mesh).astype(int)

        x_index = np.clip(x_index, 0, n - 1)
        y_index = np.clip(y_index, 0, m - 1)
        print("x_index:"+str(x_index.shape))
        print("y_index:"+str(y_index.shape))
        return y_index, x_index



    def on_EVENT_LBUTTONDOWN(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            global img_copy
            xy = "%d,%d" % (x, y)
            self.points.append((x,y))
            cv2.circle(img_copy, (x, y), 1, (255, 0, 0), thickness = -1)
            cv2.putText(img_copy, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0,0,0), thickness = 1)

            if self.count == 1:
                x1 = self.points[-2][0]
                y1 = self.points[-2][1]
                self.upleft_x = x1
                self.upleft_y = y1
                self.downright_x = x
                self.downright_y = y
                cv2.rectangle(img_copy,(x1,y1),(x,y),(255,255,255))

                self.count = 0
                self.points.clear()
                # cv2.imshow("Crop before calibration", self.img)
                imp.img = im[imp.upleft_y:imp.downright_y+1, imp.upleft_x:imp.downright_x+1]
                cv2.imshow("Crop Result", imp.img)
                cv2.waitKey(0)
                cv2.destroyWindow("Crop Result")
                img_copy = np.copy(im)

            else:
                self.count += 1
                # cv2.imshow("Crop before calibration", self.img)

    def create_dmask(self, pad=5):
        self.dmask = self.defect_mask(pad)

if __name__ == "__main__":

    f = open("config.yaml",'r+',encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)

    camid = cfg['camid']
    data_path = cfg["data_path"]
    calibration = cfg['calibration']
    method = calibration['method']
    crop = cfg['crop']

    if method == 'camera':
        print("Calibrating camera...")
        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

        # 获取标定板角点的位置
        objp = np.zeros((6 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

        obj_points = []  # 存储3D点
        img_points = []  # 存储2D点

        images = glob.glob(f"./{data_path}/sample_*.jpg")
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

            if ret:

                obj_points.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                #print(corners2)
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

                cv2.drawChessboardCorners(img, (8, 6), corners, ret)  

        # 标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
        calibration['dist'] = dist.tolist()
        calibration['mtx'] = mtx.tolist()

        img = cv2.imread(f"./{data_path}/ref.jpg")
        h, w = img.shape[:2]
        print(h,w)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#显示更大范围的图片（正常重映射之后会删掉一部分图像）
        # print (newcameramtx)
        # print("------------------使用undistort函数-------------------")
        dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
        x,y,w,h = roi
        dst1 = dst[y:y+h,x:x+w]
        cv2.imshow('calibration result of ./data_path/ref.jpg', dst1)
        cv2.waitKey(0)
        cv2.destroyWindow('calibration result of ./data_path/ref.jpg')

        print("Camera calibration finished.")

    imp = imp()
    im = cv2.imread(f'{data_path}/ref.jpg')
    img_copy = np.copy(im)

    if crop == True:
        print("Image cropping begin...")
        crop_window_name = 'Crop configuration'
        cv2.namedWindow(crop_window_name)
        cv2.setMouseCallback(crop_window_name, imp.on_EVENT_LBUTTONDOWN)
        if method == 'camera':
            img = im
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#显示更大范围的图片（正常重映射之后会删掉一部分图像）
            dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
            x,y,w,h = roi
            im = dst[y:y+h,x:x+w]
        imp.img = im
        imp.upleft_y = 0
        imp.upleft_x = 0
        imp.downright_y, imp.downright_x = np.shape(im)[:-1]
        while True:
            cv2.imshow(crop_window_name, img_copy)
            calibration['upleft_y'] = imp.upleft_y
            calibration['downright_y'] = imp.downright_y
            calibration['upleft_x'] = imp.upleft_x
            calibration['downright_x'] = imp.downright_x
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                cv2.destroyWindow(crop_window_name)
                break

        print("Image cropping finished.")

    else:
        imp.upleft_y = 0
        imp.upleft_x = 0
        imp.downright_y, imp.downright_x = np.shape(im)[:-1]
        print("No need to crop. Reset the config.yaml")
    
    cfg['calibration'] = calibration

    with open("config.yaml",'w',encoding='utf-8') as w_f:
        yaml.dump(cfg,w_f)

    if method == 'marker':
        print("Begin calibration using markers...")
        im = im[imp.upleft_y:imp.downright_y+1, imp.upleft_x:imp.downright_x+1]
        imp.img = im
        cv2.imshow("ref",im)
        cv2.waitKey(0)
        imp.create_dmask()
        m, n, c = im.shape
        # print(m,n,c)
        mask = imp.mask_marker(im)
        keypoints = imp.find_dots(mask)
        cv2.imshow("mask",mask)
        cv2.drawKeypoints(imp.img,keypoints,imp.img)
        cv2.imshow("keypoints",imp.img)
        init_array = imp.get_sortedarray(im, keypoints, False)

        corr_array = imp.get_corrarray(init_array)
        print("init array shape", np.shape(init_array))

        init_array = imp.convert_format(init_array)
        corr_array = imp.convert_format(corr_array)
        print("init array shape after", np.shape(init_array))
        x_mesh, y_mesh = np.meshgrid(range(n), range(m))
        x_index, y_index = imp.interp(corr_array, init_array, x_mesh, y_mesh)
        # print(f"x_index is {x_index.shape} and y_index is {y_index.shape}")
        abe_path = cfg['heightmap']['abe_path']
        np.savez(abe_path, x=x_index, y=y_index)
        Test = True
        if Test:
            ab_array = np.load(abe_path)
            # print("npload:" + str(np.shape(ab_array)))
            x_index = ab_array['x']
            y_index = ab_array['y']
            im_test = im #cv2.imread('test_data/ref.jpg')
            im_new = im_test[x_index, y_index, :]


            cv2.imshow('new_img', im_new)
            cv2.imshow('old_img', im_test)
            cv2.waitKey(0)
        print("Calibration using markers finished.")
