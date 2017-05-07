# utility functions
import numpy as np
import cv2
import glob
import math
import matplotlib.pyplot as plt

def loadImage(fname, greyscale=False):
    img = cv2.imread(fname)
    if(greyscale):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = (img.shape[0], img.shape[1])
    return [img, img_size]

def show_images_in_folder(img_operator, foldername='./', glob_param='*', draw_points=False, pts=[]):
    # Make a list of images
    image_filenames = glob.glob(foldername+glob_param)
    num_cols = 2
    num_rows = math.ceil(len(image_filenames)/num_cols)

    plt.rcParams['figure.figsize'] = (20.0, 20.0)

    f, axarr = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)
    f.subplots_adjust(hspace=0.1)

    for row in np.arange(0,num_rows):
        ind=0
        for fname in image_filenames[row*num_cols:np.min([len(image_filenames),row*num_cols+num_cols])]:
            img, _ = loadImage(fname, greyscale=False)
            if draw_points == True:
              img = drawPoints(img, pts)
            final_img = img_operator(img)
            axarr[row, ind].imshow(final_img, cmap='gray')
            axarr[row, ind].set_title(fname)
            ind+=1

def greyscale(img):
    tbr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return tbr

def showImage(img, figure_num=0, cmap='gray'):
    f = plt.figure(figure_num)
    plt.imshow(img, cmap='gray')
    return figure_num+1
  
def normalize(arr):
    return (arr - np.mean(arr)) / np.std(arr)

def drawLines(img, center, min_row, max_height, width, c=[0, 255, 0]):
    im_shape = img.shape
    #center = round(im_shape[1] / 2)
    min_line = [(0, min_row), (im_shape[1], min_row)]
    max_line = [(0, min_row-max_height), (im_shape[1], min_row-max_height)]
    min_w_line = [(center-width, 0), (center-width, im_shape[1])]
    max_w_line = [(center+width, 0), (center+width, im_shape[1])]
    cv2.line(img, min_line[0], min_line[1], color=c, thickness = 5)
    cv2.line(img, max_line[0], max_line[1], color=c, thickness = 5)
    cv2.line(img, min_w_line[0], min_w_line[1], color=c, thickness = 5)
    cv2.line(img, max_w_line[0], max_w_line[1], color=c, thickness = 5)

def drawBox(img, center, min_row, height, width, c=[0, 255, 0], thickness=10):
    tl = (int(center-width/2), (min_row-height))
    br = (int(center+width/2), min_row)
    cv2.rectangle(img, pt1=tl, pt2=br, color=c, thickness=thickness)

def drawPoints(img, points, radius=5, color=[0, 0, 255]):
    new_img = np.copy(img)
    for p in points:
        cv2.circle(new_img, tuple(p), radius, color, thickness=3)
    return new_img

def threshold(img, thresh, to_val=1):
    # Apply thresholding
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) & (img <= thresh[1])] = to_val
    
    return binary
    
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


