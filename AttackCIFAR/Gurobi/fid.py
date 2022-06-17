import numpy
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from scipy.linalg import sqrtm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
"""
To supress the tensorflow warnings. 
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
"""
Setting verbosity of tensorflow to minimum.
"""
from PIL import Image
from numpy import asarray
  

def loadModel():
    model = tf.keras.models.load_model(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/Models/mnist.h5')
    return model

def getImages():
    originalImages = []
    adversarialImages = []
    count = 0
    im1 = []
    im2 = []
    for path in os.listdir("/home/tooba/Documents/DNNVerification/AttackMNIST/Gurobi/AdversarialImages2"):
        originalImages.append('OriginalImages2/'+str(path))
        adversarialImages.append('AdversarialImages2/'+str(path))
        count = count+1
    
    for i in range(count):
        img = Image.open(originalImages[i])
        img = img.resize((28, 28))
        numpydata = asarray(img)
        im1.append(numpydata)

        img = Image.open(adversarialImages[i])
        img = img.resize((28, 28))
        numpydata = asarray(img)
        im2.append(numpydata)

    im1 = np.array(im1)
    im2 = np.array(im2)

    im1 = im1.reshape((im1.shape[0], 28*28)).astype('float32')
    im2 = im2.reshape((im2.shape[0], 28*28)).astype('float32')

    im1 = im1/255
    im2 = im2/255

    return im1, im2

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

model = loadModel()
im1, im2 = getImages()

fid = calculate_fid(model, im1, im2)
print(f'FID for {len(im1)} images is: {fid:.3f}')