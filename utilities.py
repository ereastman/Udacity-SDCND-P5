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

def showImage(img, figure_num=0, title=None):
    f = plt.figure(figure_num)
    plt.imshow(img)
    if title:
        plt.title(title)
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
        
def perturb(image, keep=0, angle_limit=15, scale_limit=0.1, translate_limit=3, distort_limit=3, illumin_limit=0.2):

    if(np.random.uniform() < keep):
        return image
    (W, H, C) = image.shape
    center = np.array([W / 2., H / 2.])
    da = np.random.uniform(low=-1, high=1) * angle_limit/180. * math.pi
    scale = np.random.uniform(low=-1, high=1) * scale_limit + 1

    # Use small angle approximation instead of sin/cos functions
    cc = scale*(1 - (da*da)/2.)
    ss = scale*da
    rotation    = np.array([[cc, ss],[-ss,cc]])
    translation = np.random.uniform(low=-1, high=1, size=(1,2)) * translate_limit
    distort     = np.random.standard_normal(size=(4,2)) * distort_limit

    pts1 = np.array([[0., 0.], [0., H], [W, H], [W, 0.]])
    pts2 = np.matmul(pts1-center, rotation) + center  + translation

    #add perspective noise
    pts2 = pts2 + distort

    #http://milindapro.blogspot.jp/2015/05/opencv-filters-copymakeborder.html
    matrix  = cv2.getPerspectiveTransform(pts1.astype(np.float32), pts2.astype(np.float32)) 
    perturb = cv2.warpPerspective(image, matrix, (W, H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)  # BORDER_WRAP  #BORDER_REFLECT_101  #cv2.BORDER_CONSTANT  BORDER_REPLICATE

    #brightness, contrast, saturation-------------
    #from mxnet code
    if 1:  #brightness
        alpha = 1.0 + illumin_limit*np.random.uniform(-1, 1)
        #alpha = 1.0 + illumin_limit*-1
        perturb = perturb * alpha
        perturb = np.clip(perturb,0.,255.)
        pass

    coef = np.array([[[0.299, 0.587, 0.114]]]) #rgb to gray (YCbCr) :  Y = 0.299R + 0.587G + 0.114B

    if 1:  #contrast
        alpha = illumin_limit*np.random.uniform(-1, 1)
        #alpha = illumin_limit*-1
        gray = perturb * coef
        gray = (3.0 * (alpha) / gray.size) * np.sum(gray)
        perturb = perturb * (1.0 + alpha)
        perturb += gray
        perturb = np.clip(perturb,0.,255.)
        pass

    if 1:  #saturation
        alpha = illumin_limit*np.random.uniform(-1, 1)
        #alpha = illumin_limit*-1
        gray = perturb * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        #print(gray.shape)
        #print(alpha.shape)
        gray = np.multiply(alpha, gray)
        perturb = perturb * (1.0 + alpha)
        perturb += gray
        perturb = np.clip(perturb,0.,255.)
        pass

    return perturb
