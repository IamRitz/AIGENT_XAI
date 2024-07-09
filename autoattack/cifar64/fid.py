import numpy
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from PIL import Image
from numpy import asarray
from cmath import log

def calculate_fid(model, images1, images2):
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid

def PielouMeaure(frequencies, num_classes):
    sum = 0
    for i in range(num_classes):
        sum=sum+frequencies[i]
    
    percents = []
    for i in range(num_classes):
        percents.append(float(frequencies[i])/sum)

    # print(percents)
    measure = 0
    for i in range(num_classes):
        if percents[i]==0:
            measure = measure+percents[i]*log(percents[i]+1)
        else:
            measure = measure+percents[i]*log(percents[i])
    return -1*measure/log(num_classes)

def calculate_ps(model, images1, images2):
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    ps = [0]*len(act2[0])
    success = 0

    for i in range(len(act2)):
        label_o = np.argmax(act1[i])
        label = np.argmax(act2[i])
        ps[label] = ps[label] + 1
        if label_o!=label:
            success += 1
    
    # print(ps)
    return PielouMeaure(ps, len(act2[0])), success
