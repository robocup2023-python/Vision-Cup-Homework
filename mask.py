import cv2
import numpy as np
img1=cv2.imread("./samoyed.jpeg",0)
img2=cv2.imread("./rec.jpg",0)
res = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
# 保存
# cv2.imencode('.jpg', res)[1].tofile(r'D:\360MoveData\Users\Administrator\Desktop\wo\add_img.jpg')
cv2.imshow('input_image', res)
cv2.waitKey(0)
