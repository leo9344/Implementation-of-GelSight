from mimetypes import init
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
import yaml
import argparse

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

    def load_crop_params(self):
        crop_coordinates = np.loadtxt("crop_coordinates.txt",dtype=np.uint16)
        [[self.upleft_x, self.upleft_y],[self.downright_x, self.downright_y]] = crop_coordinates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-c or --crop for crop image.")
    parser.add_argument("-c", "--crop", action="store_true",help='-c or --crop for cropping the image.')
    args = parser.parse_args()
    crop = args.crop

    f = open("config.yaml",'r',encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)

    camid = cfg['camid']
    data_path = cfg["data_path"]
    calibration = cfg['calibration']
    # image processing class
    imp = imp()

    # read a non-contact image
    im = cv2.imread(f'{data_path}/ref.jpg')
    img_copy = np.copy(im)

    if crop == True:
        calibration['crop'] = True
        cv2.namedWindow("Crop before calibration")
        cv2.setMouseCallback("Crop before calibration", imp.on_EVENT_LBUTTONDOWN)
        imp.img = im
        imp.upleft_y = 0
        imp.upleft_x = 0
        imp.downright_y, imp.downright_x = np.shape(im)[:-1]
        while True:
            cv2.imshow("Crop before calibration", img_copy)
            calibration['upleft_y'] = imp.upleft_y
            calibration['downright_y'] = imp.downright_y
            calibration['upleft_x'] = imp.upleft_x
            calibration['downright_x'] = imp.downright_x
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow("Crop before calibration")
                break

        print("Crop")
    else:
        imp.load_crop_params()
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
    np.savez('abe_corr.npz', x=x_index, y=y_index)

    crop_coordinates = [[imp.upleft_x, imp.upleft_y],[imp.downright_x, imp.downright_y]]
    crop_coordinates = np.array(crop_coordinates,dtype=np.uint16)
    np.savetxt("crop_coordinates.txt",crop_coordinates)
    # load the x_index and y_index if not for calibration
    # Test the new image
    Test = True
    if Test:
        ab_array = np.load('abe_corr.npz')
        # print("npload:" + str(np.shape(ab_array)))
        x_index = ab_array['x']
        y_index = ab_array['y']
        im_test = im #cv2.imread('test_data/ref.jpg')
        im_new = im_test[x_index, y_index, :]


        cv2.imshow('new_img', im_new)
        cv2.imshow('old_img', im_test)
        cv2.waitKey(0)











