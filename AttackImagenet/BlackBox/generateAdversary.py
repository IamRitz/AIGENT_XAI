from time import time
from pielouMeasure import PielouMeaure
from FID import calculate_fid, getImages
from tensorflow.keras.utils import to_categorical

import numpy as np
import tensorflow as tf
from PIL import Image
import numpy as np
import random
print("HEERREE")
from prepareData import getTrainData, getValData

def getData(w, h):
    X_train, y_train = getTrainData(w, h)
    X_test, y_test = getValData(w, h)
    print(X_train.shape)
    print(y_train.shape)
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train, X_test, y_test

def getModel():
    w, h = 128, 128
    print("[INFO] loading dataset...")
    X_train, y_train, X_test, y_test = getData(w, h)
    trainX = np.expand_dims(X_train, axis=-1)
    testX = np.expand_dims(X_test, axis=-1)
    trainY = to_categorical(y_train, 10)
    testY = to_categorical(y_test, 10)
    model = tf.keras.models.load_model('imagenette_3.h5')
    return model, testX, testY

def generate_image_adversary(image):
#   print(np.shape(image))
  w, h = 128, 128
  image_array = np.array(image).reshape(1, w*h*3)
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
  image = image_array.reshape(1, w, h, 3)
  return image

def convertToMtarix(array, m, n, channels):
    for i in range(m*n*channels):
        array[i] = 255*array[i]
    matrix = np.array(array)
    return matrix.reshape((m, n, channels))

def show(pixelMatrix, w, h, channels):
    # print(np.shape(pixelMatrix))
    pixelMatrix = convertToMtarix(pixelMatrix[0], w, h, channels)
    data = np.array(pixelMatrix)
    im = Image.fromarray(data.astype(np.uint8), mode='RGB')
    # im.show()
    return im

def generateAdversarial(model, testX, testY, folderCount):
    w, h = 128, 128
    countOriginal = [0]*10
    countAdversary = [0]*10
    avConf = 0
    totalL2 = 0
    totalLinf = 0
    avCount = 0
    success = 0

    for i in range(400):
        image = testX[i]
        original_array = np.array(image).reshape(1, w*h*3)
        label = testY[i]
        adversary = generate_image_adversary(image.reshape(1, w, h, 3))
        pred = model.predict(np.array(adversary).reshape(1, w*h*3))
        adversary = adversary.reshape( w, h, 3)
        adversarial_array = np.array(adversary).reshape(1, w*h*3)
        
        l2 = np.linalg.norm(np.array(adversarial_array)-np.array(original_array))
        diff = np.array(adversarial_array[0])-np.array(original_array[0])
        diff = [abs(x) for x in diff]

        countChange = 0
        for x in diff:
            if x!=0:
                countChange = countChange + 1

        linf = np.argmax(diff)
        originalImage = show(original_array, w, h, 3)
        
        adversarialImage = show(adversarial_array, w, h, 3)
        originalLabel = np.argmax(label)
        adversaryLabel = np.argmax(pred)
        if originalLabel!=adversaryLabel:
            success = success + 1
        else:
            continue
        if countOriginal[originalLabel]<30:
            countAdversary[adversaryLabel] = 1 + countAdversary[adversaryLabel]
            countOriginal[originalLabel] = 1 + countOriginal[originalLabel]
        
        originalImage.save("OriginalImages/Image_"+str(i)+".jpg")
        adversarialImage.save("AdversarialImages/Image_"+str(i)+".jpg")

        # originalImage = np.dstack([originalImage])
        # cv2_imshow(originalImage)

        # adversarialImage = np.dstack([adversarialImage])
        # cv2_imshow(adversarialImage)
        # break

        print("Original Label =", np.argmax(label), "\nAdversarial Label=", np.argmax(pred), "\nL2-distance=", l2, "\nL-infinity-distance=", diff[int(linf)], "\nNumber of pixels changed=", countChange)
        totalL2 = totalL2 + l2
        totalLinf = totalLinf + diff[int(linf)]
        avCount = avCount + countChange
        print(pred[0][adversaryLabel])
        avConf = avConf + pred[0][adversaryLabel]
        print("..........................................")
    return avConf, totalL2, totalLinf, avCount, success, countOriginal, countAdversary

def displayResults():
    folderCount = ""

    model, textX, textY = getModel()
    t1 = time()
    avConf, totalL2, totalLinf, avCount, success, countOriginal, countAdversary = generateAdversarial(model, textX, textY, folderCount)
    t2 = time()
    print("Average Confidence:",avConf/success)
    print("Average L2:",totalL2/success)
    print("Average L-inf:",totalLinf/success)
    print("Average number of pixels modified:", avCount/success)
    print("Percentage of pixels modified(average):", (avCount/success)*100/(32*32*3)," %")
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

print("HEERREE")
displayResults()