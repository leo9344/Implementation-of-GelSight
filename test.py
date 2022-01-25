from mimetypes import init
import cv2
import numpy as np
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
        # cv2.waitKey(0)
        # cv2.waitKey(1)
        mask = cv2.resize(mask.astype(np.uint8), (m, n))
        mask = cv2.dilate(mask, self.kernel, iterations=1) * self.dmask
        cv2.imshow('mask', mask.astype(np.uint8) * 255)
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
        # print(f"keypoint size is {len(keypoints)}")
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
                    number = int(input(f"Time {i} Enter the number of misclassified point: "))
                    # number = 0
                    temp_new = []
                    while number > 0:
                        print(1)
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

        # print(f"temp size is {np.array(xy_array).shape}")
        # print(f"xy array is {np.array(xy_array)}")
        return xy_array
    def create_dmask(self, pad=5):
        self.dmask = self.defect_mask(pad)

imp = imp()
cap = cv2.VideoCapture(0)
while(1):
    ret,frame = cap.read()
    frame = frame[imp.upleft_y:imp.downright_y+1, imp.upleft_x:imp.downright_x+1]
    imp.img = frame
    imp.create_dmask()
    mask = imp.mask_marker(frame)
    keypoints = imp.find_dots(mask)
    cv2.imshow("mask",mask)
    cv2.drawKeypoints(imp.img,keypoints,imp.img)
    cv2.imshow("keypoints",imp.img)
    init_array = imp.get_sortedarray(frame, keypoints, False)

    count = 1
    for i in range(5):
        for j in range(5):
            x,y = init_array[i][j]
            x = int(x)
            y = int(y)
            cv2.putText(frame,f"{count}",(x,y),cv2.FONT_HERSHEY_PLAIN,
                            1.0, (0,0,0), thickness = 1)
            count += 1
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
