import numpy as np
import cv2

# DRAWING FUNCTIONS

def comp_triangle(q, p):
    v1 = p
    v2 = np.array([-p[1], p[0]])
    e1 = v1/np.linalg.norm(v1)*5
    e2 = v2/np.linalg.norm(v2)*5
    p1 = q + 2*e1
    p2 = q + 0.7*e2
    p3 = q - 0.7*e2
    return p1, p2, p3
  
def draw_triangle(img, p1, p2, p3, ratio, color=(255, 0, 0), center=(50, 50)):
    p1 = p1-center+250
    p2 = p2-center+250
    p3 = p3-center+250
    cv2.line(img, (int(p1[0]*ratio), int(p1[1]*ratio)), (int(p2[0]*ratio), int(p2[1]*ratio)), color, 1) 
    cv2.line(img, (int(p2[0]*ratio), int(p2[1]*ratio)), (int(p3[0]*ratio), int(p3[1]*ratio)), color, 1) 
    cv2.line(img, (int(p1[0]*ratio), int(p1[1]*ratio)), (int(p3[0]*ratio), int(p3[1]*ratio)), color, 1) 

def write_image(img, text, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    cv2.putText(img, text, pos, font, fontScale=0.3, color=(0, 0, 0), thickness=1)

def draw_axis(img, center, size=150, im_pos=250):
    x, y = int(center[0]), int(center[1])
    cv2.line(img, (im_pos-size,im_pos-size), (im_pos+size, im_pos-size), color=(0,0,0))
    cv2.line(img, (im_pos-size,im_pos-size), (im_pos-size, im_pos+size), color=(0,0,0))
    #cv2.circle(img, (im_pos,im_pos), 1, (0,0,0), thickness=2)
    write_image(img, f"{x-size}", (im_pos-size, im_pos-size-5))
    write_image(img, f"{y-size}", (im_pos-size-20, im_pos-size+10))
    for i in range(1,2*size//50+1):
        write_image(img, f"{x-size+i*50}", (im_pos-size+i*50, im_pos-size-5))
        write_image(img, f"{y-size+i*50}", (im_pos-size-20, im_pos-size+i*50))

def draw_circle(img, yk, Rk, center):
    cv2.circle(img, (int(yk[0]-center[0]+250), int(yk[1]-center[1]+250)), int(Rk), (0,0,0), thickness=2)