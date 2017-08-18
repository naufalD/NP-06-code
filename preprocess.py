import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob, os, time, cv2

basedir = r"C:\Projects\Programming\Retina"

def scaleRadius(img, scale):
    x = img[int(img.shape[0]/2), :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale *1.0/r
    return cv2.resize(img, (0, 0), fx=s, fy=s)

def graham_preprocess(image):
    image = scaleRadius(image, scale)
    # subtract local mean color
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), scale / 30), -4, 128)
    # remove outer 10%
    b = np.zeros(image2.shape)
    cv2.circle(b, (int(image.shape[1]/2), int(image.shape[0]/2)), 500, (1, 1, 1), -1, 8, 0)
    image = image * b + 128*(1-b)
    return image

def get_data(filename):
    img_dir = basedir +'\\Reshaped_test\\'+ filename
    print(img_dir)
    image = plt.imread(img_dir)
    return image

def preprocess(image):
    border = int((image.shape[1]-image.shape[0])/2)
    image = image[:,border:image.shape[1]-border,:]
    cv2.imwrite(r"C:\Projects\Programming\Retina\Presentation\left_10_cropped.jpeg", image)
    image = cv2.resize(image, (512,512))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 17), -4, 128)
    cv2.imwrite(r"C:\Projects\Programming\Retina\Presentation\left_10_normalised.jpeg", image)
    b = np.zeros(image.shape)
    cv2.circle(b, (int(image.shape[1]/2), int(image.shape[0]/2)), 225, (1, 1, 1), -1, 8, 0)
    image = image * b
    cv2.imwrite(r"C:\Projects\Programming\Retina\Presentation\left_10_circled.jpeg", image)
    return image

#dirlist = os.listdir(basedir +'\\test\\')

#for x in range(0, 53576):
#    filename = dirlist[x]
#    image = get_data(filename)
#    print (filename, x)
#    image = cv2.resize(image, (256,256))
#    new_directory = basedir+'\\Reshaped_test_256\\'+filename
#    print (new_directory)
#    cv2.imwrite(new_directory, image)
#dirlist = os.listdir(basedir +'\\test\\')
#print(dirlist[51972])
