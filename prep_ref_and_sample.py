from ast import parse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os
import sys

def cap_ref(event, x, y, flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.imwrite(f"{data_path}/ref.jpg",frame)
        print(f"{data_path}/ref.jpg captured. Press 'Q' to exit.")

def cap_sample(event, x, y, flags,params):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.imwrite(f"{data_path}/sample_{count}.jpg",frame)
        print(f"{data_path}/sample_{count}.jpg captured.")
        count += 1 




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture ref.jpg and sample_xx.jpg')
    parser.add_argument("-r", "--ref", action="store_true", help='-r or --ref for capturing ref.jpg')
    parser.add_argument("-s", "--sample", action='store_true', help='-s or --sample for capturing sample_xx.jpg')

    args = parser.parse_args()

    ref = args.ref
    sample = args.sample

    if ref == False and sample == False:
        print("Nothing happened.")
        sys.exit(0)
    
    f = open("config.yaml",'r',encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)

    camid = cfg['camid']
    sample_from, sample_to = cfg['sample']['from'], cfg['sample']['to']
    data_path = cfg['data_path']

    count = sample_from

    cap = cv2.VideoCapture(camid)

    if ref == True:
        cv2.namedWindow("capture ref")
        cv2.setMouseCallback("capture ref", cap_ref)
        while(1):
            ret, frame = cap.read()
            cv2.imshow("capture ref", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow("capture ref")
                break
    
    
    if sample == True:
        cv2.namedWindow("capture sample")
        cv2.setMouseCallback("capture sample", cap_sample)
        while(1):
            ret, frame = cap.read()
            cv2.imshow("capture sample", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or count > sample_to:
                cv2.destroyWindow("capture sample")
                break
    
    cap.release()
    cv2.destroyAllWindows() 