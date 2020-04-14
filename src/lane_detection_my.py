import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from matplotlib import pyplot as plt
Blur_size = 5

#canny算子
threshold1 = 50
threshold2 = 150

# Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 40
max_line_gap = 20

def roi_mask(edges, roi_vtx):
    mask = np.zeros_like(edges)
    if len(edges.shape) > 2:
        channel_out = edges.shape[2]
        mask_color = (255,) * channel_out
    else:
        mask_color = 255
    cv2.fillPoly(mask, roi_vtx, mask_color)
    mask_image = cv2.bitwise_and(edges, mask)
    plt.imshow(mask_image)
    plt.show()
    return mask_image
    pass

def draw_lanes(line_img, lines,color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    plt.imshow(line_img)
    plt.show()
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    draw_lanes(line_img, lines)
    return line_img

def process_an_image(img):
    roi_vtx = np.array([[(0, img.shape[0]), (460, 325), (520, 325), (img.shape[1], img.shape[0])]])
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray_img, (Blur_size, Blur_size), 0, 0)
    edges = cv2.Canny(blur_gray, threshold1, threshold2)

    plt.imshow(edges)
    plt.axis('off')
    plt.show()
    roi_edges = roi_mask(edges, roi_vtx)
    line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)
    res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    plt.imshow(res_img)
    plt.show()
    return res_img

if __name__ == '__main__':
    output = '../resources/video_1_sol.mp4'
    clip = VideoFileClip("../resources/video_1.mp4")
    
    out_clip = clip.fl_image(process_an_image)
    #out_clip.write_videofile(output, audio=False)