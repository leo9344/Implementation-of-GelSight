import cv2
img = cv2.imread("./asset/tracking1.png")
img2 = cv2.imread("./asset/tracking2.png")
cv2.imwrite("./asset/tracking1.jpg",img)
cv2.imwrite("./asset/tracking2.jpg",img2)