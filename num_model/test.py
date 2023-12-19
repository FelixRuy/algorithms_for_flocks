import numpy as np
import cv2


frame_height = 500
frame_width = 1000
img = np.ones((frame_height, frame_width, 3), np.uint8) * 255

def draw_triangle(img, p1, p2, p3, color=(255, 0, 0)):
    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1) 
    cv2.line(img, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), color, 1) 
    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p3[0]), int(p3[1])), color, 1) 

q = np.array([250, 250])
p = np.array([100, 100])

v1 = p
v2 = np.array([-p[0], p[0]])
u1 = v1
u2 = v2 - u1@v2/u1@u1 * u1
e1 = u1/np.linalg.norm(u1)*5
e2 = u2/np.linalg.norm(u2)*5

p1 = q + 2*e1
p2 = q + 0.7*e2
p3 = q - 0.7*e2

print(p1, p2, p3)

draw_triangle(img, p1, p2, p3)

cv2.imshow("FLOCK IT", img) 

cv2.waitKey(0) 