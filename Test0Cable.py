import cv2
import time
import numpy as np
from fast_poisson import fast_poisson
from scipy.interpolate import griddata
import yaml
from cam_preprocessing import get_image
def make_kernel(n, k_type):
    if k_type == 'circle':
        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    else:
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))
    return kernal


f = open("config.yaml",'r+',encoding='utf-8')
cfg = yaml.load(f, Loader=yaml.FullLoader)

camid = cfg['camid']
data_path = cfg["data_path"]
calibration_cfg = cfg['calibration']
heightmap = cfg['heightmap']

img_counter1 = 0
table = np.load(heightmap['table_smooth_path'])
red_range = calibration_cfg['red_range']
ratio = calibration_cfg['ratio']
bin_num = int((red_range[1] - red_range[0])*ratio)
zeropoint = calibration_cfg['zeropoint']
lookscale = calibration_cfg['lookscale']
# abe_array = np.load('abe_corr.npz') # change this with your aberration array
# x_index = abe_array['x']
# y_index = abe_array['y']

# crop_coordinates = np.loadtxt("crop_coordinates.txt",dtype=np.uint16)
downright_x = calibration_cfg['downright_x']
downright_y = calibration_cfg['downright_y']
upleft_x = calibration_cfg['upleft_x']
upleft_y = calibration_cfg['upleft_y']


# print("x_index:",np.array(x_index).shape)
# print("y_index:",np.array(y_index).shape)
pad = 5
kernel2 = make_kernel(9, 'circle')
con_flag1 = True
reset_shape1 = True
restart1 = False

y, x = 15, 20


def matching_v2(test_img, ref_blur, blur_inverse):
    diff_temp1 = test_img - ref_blur
    diff_temp2 = diff_temp1 * blur_inverse
    diff_temp3 = np.clip((diff_temp2 - zeropoint) / lookscale, 0,
                         0.999)
    # diff_temp3 = diff_temp2
    print(np.shape(diff_temp2), np.shape(diff_temp3))
    diff = (diff_temp3 * bin_num).astype(int)
    grad_img = table[diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], :]
    return grad_img

def defect_mask(img):
    pad = 5
    var0 = 60  # left up
    var1 = 60  # right up
    var2 = 65  # right down
    var3 = 60  # left down
    im_mask = np.ones((img.shape))
    # triangle0 = np.array([[0, 0], [var0, 0], [0, var0]])
    # triangle1 = np.array([[im_mask.shape[1] - var1, 0],
    #                       [im_mask.shape[1], 0], [im_mask.shape[1], var1]])
    # triangle2 = np.array([[im_mask.shape[1] - var2, im_mask.shape[0]], [im_mask.shape[1], im_mask.shape[0]], \
    #     [im_mask.shape[1], im_mask.shape[0]-var2]])
    # triangle3 = np.array([[0, im_mask.shape[0]],
    #                       [0, im_mask.shape[0] - var3],
    #                       [var3, im_mask.shape[0]]])
    # color = [0]  #im_mask
    # cv2.fillConvexPoly(im_mask, triangle0, color)
    # cv2.fillConvexPoly(im_mask, triangle1, color)
    # cv2.fillConvexPoly(im_mask, triangle2, color)
    # cv2.fillConvexPoly(im_mask, triangle3, color)
    im_mask[:pad, :] = 0
    im_mask[-pad:, :] = 0
    im_mask[:, :pad * 2 + 20] = 0
    im_mask[:, -pad:] = 0
    return im_mask.astype(int)

def make_mask(img, keypoints):
    img = np.zeros_like(img[:, :, 0])
    for i in range(len(keypoints)):
        # cv2.circle(img, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), 6, (1), -1)
        cv2.ellipse(img,
                    (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])),
                    (9, 6), 0, 0, 360, (1), -1)

    return img

def find_dots(binary_image):
    # down_image = cv2.resize(binary_image, None, fx=2, fy=2)
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 12
    params.minDistBetweenBlobs = 30
    params.filterByArea = True
    params.minArea = 9
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minInertiaRatio = 0.5
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(binary_image.astype(np.uint8))

    return keypoints

def marker_detection(raw_image):
    m, n = raw_image.shape[1], raw_image.shape[0]
    # raw_image = cv2.pyrDown(raw_image).astype(np.float32)
    raw_image_blur = cv2.GaussianBlur(raw_image.astype(np.float32), (5, 5),
                                      0)
    ref_blur = cv2.GaussianBlur(raw_image.astype(np.float32), (25, 25), 0)
    diff = ref_blur - raw_image_blur
    diff *= 16.0
    diff[diff < 0.] = 0.
    diff[diff > 255.] = 255.
    mask = ((diff[:, :, 0] > 25) & (diff[:, :, 2] > 25) &
            (diff[:, :, 1] > 120))
    mask = cv2.resize(mask.astype(np.uint8), (m, n))
    mask = cv2.dilate(mask, kernel2, iterations=1)
    return mask

def crop_image(img, pad):
    return img[pad:-pad, pad:-pad]

# cap3 = cv2.VideoCapture("http://192.168.3.8:9000/stream.mjpg")
print("[INFO] warming up webcam...")

# cap1 = cv2.VideoCapture("http://192.168.137.28:9000/stream.mjpg")
# cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1 = cv2.VideoCapture(camid)

# cap1.set(3, 320)
# cap1.set(4, 240)


# def mouse_click(event, x, y, flags, para):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # print('PIX:', x, y)
#         print("BGR:", frame1[y, x])
#         # print("Image Size", frame1.shape[0], frame1.shape[1])
#
# cv2.namedWindow("PiCam1")
# cv2.setMouseCallback("PiCam1", mouse_click)

count = 0

from Visualizer3D import Visualizer
# print(downright_x, upleft_x, upleft_y , downright_y)
vis = Visualizer(downright_x-upleft_x-pad*2+1, downright_y-upleft_y-pad*2+1)
# from scipy.interpolate import griddata
def dilate(img, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)
    
def interpolate_grad(img, mask):
    # mask = (soft_mask > 0.5).astype(np.uint8) * 255
    # cv2.imshow("mask_hard", mask)
    # pixel around markers
    mask_around = (dilate(mask, ksize=3) > 0) & (mask != 1)
    # mask_around = mask == 0
    mask_around = mask_around.astype(np.uint8)

    x, y = np.arange(img.shape[0]), np.arange(img.shape[1])
    yy, xx = np.meshgrid(y, x)

    # mask_zero = mask == 0
    mask_zero = mask_around == 1
    mask_x = xx[mask_zero]
    mask_y = yy[mask_zero]
    points = np.vstack([mask_x, mask_y]).T
    values = img[mask_x, mask_y]
    markers_points = np.vstack([xx[mask != 0], yy[mask != 0]]).T
    # method = "nearest"
    method = "linear"
    # method = "cubic"
    x_interp = griddata(points, values, markers_points, method=method)
    x_interp[x_interp != x_interp] = 0.0
    ret = img.copy()
    ret[mask != 0] = x_interp
    return ret

while(cap1.isOpened()):
    t = time.time()
    # ret1, frame1 = cap1.read()


    # cv2.imshow('PiCam1', frame1)
    if con_flag1:
        ret1, raw_image = cap1.read()
        # raw_image1 = imutils.resize(raw_image, width=320)
        raw_image = get_image(raw_image)
        ref_image = crop_image(raw_image, pad)

        cv2.imshow('ref_image', ref_image)

        marker = marker_detection(ref_image.copy())
        keypoints = find_dots((1 - marker) * 255)
        if reset_shape1:
            marker_mask = make_mask(ref_image.copy(), keypoints)
            ref_image = cv2.inpaint(ref_image, marker_mask, 3,
                                    cv2.INPAINT_TELEA)
            red_mask = (ref_image[:, :, 2] > 12).astype(np.uint8)
            dmask1 = defect_mask(ref_image[:, :, 0])
            ref_blur1 = cv2.GaussianBlur(ref_image.astype(np.float32),
                                              (3, 3), 0)
            blur_inverse1 = 1 + ((np.mean(ref_blur1) /
                                       (ref_blur1 + 1)) - 1) * 2

            # cv2.imshow('blur_inverse1', blur_inverse1)
        # u_addon1 = list(np.zeros(len(keypoints)))
        # v_addon1 = list(np.zeros(len(keypoints)))
        # x_iniref1 = []
        # y_iniref1 = []
        marker_num = len(keypoints)
        mp_array = np.zeros((marker_num, 3, 200))
        index_ref = np.linspace(0, marker_num - 1,
                                marker_num).astype(int)
        con_flag1 = False
        reset_shape1 = False

    else:
        if restart1:
            con_flag1 = True
            restart1 = False
        # print("now is second round")
        ret1, raw_image = cap1.read()
        raw_image = get_image(raw_image)
        raw_image2 = raw_image
        cv2.imshow('raw_image2', raw_image2)
        raw_image3 = crop_image(raw_image2, pad)
        # cv2.imshow('raw_image3', raw_image3)
        raw_image4 = cv2.GaussianBlur(raw_image3, (3, 3), 0)



        marker_mask = marker_detection(raw_image4) * dmask1
        cv2.imshow("marker_mask",marker_mask.astype(np.uint8)*255)
        interp = cv2.inpaint(raw_image4, marker_mask.astype(np.uint8), 10, cv2.INPAINT_TELEA)
        raw_image4 = interp
        # interp_b = cv2.inpaint(raw_image4[:,:,0],marker_mask.astype(np.uint8),10,cv2.INPAINT_TELEA)
        # interp_g = cv2.inpaint(raw_image4[:,:,1],marker_mask.astype(np.uint8),10,cv2.INPAINT_TELEA)
        # interp_r = cv2.inpaint(raw_image4[:,:,2],marker_mask.astype(np.uint8),10,cv2.INPAINT_TELEA)
        # # raw_image2 = cv2.inpaint(raw_image2,marker_mask.astype(np.uint8),3,cv2.INPAINT_TELEA)
        # raw_image4 = cv2.merge([interp_b,interp_g,interp_r])
        cv2.imshow("raw_image4",raw_image4)
        # print(np.shape(raw_image4),type(raw_image4[0,0,0]))
        # print("")q
        # print(len(marker_mask))
        # print(type(marker_mask[0,0]))
        # cv2.cvtColor(marker_mask,cv2.bin)
        # cv2.imshow("marker_mask",marker_mask)
        # print(marker_mask)
        # cv2.imshow("marker mask",)
        # cv2.imshow("marker_mask", marker_mask.astype(np.uint8)*255)

        grad_img2 = matching_v2(raw_image4, ref_blur1, blur_inverse1)
        # grad_img2[marker_mask==1] = 0

        # grad_img2 = interpolate_grad(grad_img2, marker_mask.astype(np.uint8))

        depth = fast_poisson(grad_img2[:, :, 0] * (1 - marker_mask) * red_mask,
                             grad_img2[:, :, 1] * (1 - marker_mask) * red_mask)
        depth[depth < 0] = 0
        # width, height = np.shape(depth)[0], np.shape(depth)[1]
        # points = []
        # values = []
        # xi = []
        # for i in range(width):
        #     for j in range(height):
        #         points.append([i,j])
        #         values.append(depth[i,j])
        #         if marker_mask[i,j]>0:
        #             xi.append([i,j])
        # interp = griddata(points,values ,xi,method='linear')
        # for i in range(len(interp)):
        #     depth[xi[i][0], xi[i][1]] = interp[i]
        # print("points",np.shape(points),"values",np.shape(values),"xi",np.shape(xi),"interp",np.shape(interp))
        
        # depth = cv2.GaussianBlur(depth,(5,5),0)
        # depth/=2
        vis.update(depth)
        # print(np.shape(depth)) (228,289)
        
        depth = cv2.applyColorMap((depth * 200).astype(np.uint8),
                                  cv2.COLORMAP_TWILIGHT)
        # cv2.imshow('camera', raw_image1)
        depth = cv2.GaussianBlur(depth,(11,11),0)
        cv2.imshow('depth', depth.astype(np.uint8) * 255)
        # print(np.shape(depth))

        cv2.waitKey(1)

    img_counter1 += 1
    # print(1 / (time.time() - t))

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    # if cv2.waitKey(20) & 0xFF == ord('s'):
    #     cv2.imwrite('Sample_' + str(count).zfill(2) + '.jpg', raw_image)
    #     count += 1
    #     print(f"Image {count} Saved!")

cap1.release()
cv2.destroyAllWindows()
