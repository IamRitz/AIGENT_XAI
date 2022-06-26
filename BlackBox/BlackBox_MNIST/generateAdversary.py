from time import time
from pielouMeasure import PielouMeaure
from FID import calculate_fid, getImages
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from PIL import Image
import numpy as np
import random

def getModel():
    print("[INFO] loading MNIST dataset...")
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX / 255.0
    testX = testX / 255.0
    trainX = np.expand_dims(trainX, axis=-1)
    testX = np.expand_dims(testX, axis=-1)
    trainY = to_categorical(trainY, 10)
    testY = to_categorical(testY, 10)
    model = tf.keras.models.load_model('mnistModel.h5')
    return model, testX, testY

def generate_image_adversary(image):
#   print(np.shape(image))
  image_array = np.array(image).reshape(1, 28*28)
  for i in range(len(image_array[0])):
    selector = random.randint(0,100)
    if selector%4==0:
      val = 30/255

    elif selector%4==1:
      val = -30/255

    if selector%4==2:
      val = 60/255

    else:
      val = -60/255
    image_array[0][i] = image_array[0][i]+val
  image = image_array.reshape(1, 28, 28, 1)
  return image

def convertToMtarix(array, m, n):
    for i in range(m*n):
        array[i] = 255*array[i]
    matrix = np.array(array)
    return matrix.reshape((m,n))

def show(pixelMatrix, w, h):
    # print(np.shape(pixelMatrix))
    pixelMatrix = convertToMtarix(pixelMatrix[0], 28, 28)
    data = np.array(pixelMatrix)
    im = Image.fromarray(data.astype(np.uint8), mode='L')
    # im = im.resize((28, 28))
    # im.show()
    return im

def generateAdversarial(model, testX, testY, folderCount):
    countOriginal = [0]*10
    countAdversary = [0]*10
    avConf = 0
    totalL2 = 0
    totalLinf = 0
    avCount = 0
    success = 0

    for i in range(500):
        image = testX[i]
        original_array = []
        for k in range(len(image)):
            for j in range(len(image[k])):
                original_array.append(image[k][j][0])

        label = testY[i]
        adversary = generate_image_adversary(image.reshape(1, 28, 28, 1))
        pred = model.predict(adversary)
        adversary = adversary.reshape(28, 28)
        adversarial_array = [j for sub in adversary for j in sub]
        
        l2 = np.linalg.norm(np.array(adversarial_array)-np.array(original_array))
        diff = np.array(adversarial_array)-np.array(original_array)
        diff = [abs(x) for x in diff]

        countChange = 0
        for x in diff:
            if x!=0:
                countChange = countChange + 1

        linf = np.argmax(diff)
        originalImage = show([original_array], 28, 28)
        
        adversarialImage = show([adversarial_array], 28, 28)
        originalLabel = np.argmax(label)
        adversaryLabel = np.argmax(pred)
        if adversaryLabel!= originalLabel:
            success = 1 + success
        else:
            continue

        if countOriginal[originalLabel]<30:
            countAdversary[adversaryLabel] = 1 + countAdversary[adversaryLabel]
            countOriginal[originalLabel] = 1 + countOriginal[originalLabel]
        

        originalImage.save("OriginalImages"+str(folderCount)+"/Image_"+str(i)+".jpg")
        adversarialImage.save("AdversarialImages"+str(folderCount)+"/Image_"+str(i)+".jpg")

        # originalImage = np.dstack([originalImage] * 3)
        # originalImage = cv2.resize(originalImage, (96, 96))
        # cv2_imshow(originalImage)

        print("Original Label =", np.argmax(label), "\nAdversarial Label=", np.argmax(pred), "\nL2-distance=", l2, "\nL-infinity-distance=", diff[int(linf)], "\nNumber of pixels changed=", countChange)
        totalL2 = totalL2 + l2
        totalLinf = totalLinf + diff[int(linf)]
        avCount = avCount + countChange
        print(pred[0][adversaryLabel])
        avConf = avConf + pred[0][adversaryLabel]
        print("..........................................")
    return avConf, totalL2, totalLinf, avCount, success, countOriginal, countAdversary

def displayResults():
    folderCount = 1

    model, textX, textY = getModel()
    t1 = time()
    avConf, totalL2, totalLinf, avCount, success, countOriginal, countAdversary = generateAdversarial(model, textX, textY, folderCount)
    t2 = time()
    print("Average Confidence:",avConf/success)
    print("Average L2:",totalL2/success)
    print("Average L-inf:",totalLinf/success)
    print("Average number of pixels modified:", avCount/success)
    print("Percentage of pixels modified(average):", (avCount/success)*100/784," %")
    print("Attack was successful was on:", success, "Images.")
    print("Average time taken: ", (t2-t1)/success, "seconds.")
    print(countOriginal)
    print(countAdversary)

    im1, im2 = getImages(folderCount)
    
    fid = calculate_fid(model, im1, im2)
    print(f'FID for {len(im1)} images is: {fid:.6f}')

    toCheck = countAdversary

    print("Pielou Meaure is: ",PielouMeaure(toCheck, len(toCheck)))
    print()

displayResults()