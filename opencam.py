import cv2
import numpy as np
mtx = np.loadtxt("./mtx.txt")
dist = np.loadtxt("./dist.txt")

cap = cv2.VideoCapture(0)
while(1):
    ret, img = cap.read()
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
    x,y,w,h = roi
    dst1 = dst[y:y+h,x:x+w]
    cv2.imshow("dst",dst1)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break
cap.release()
cv2.destroyAllWindows()



