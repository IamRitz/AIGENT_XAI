import os
from time import time
from pielouMeasure import PielouMeaure
from FID import calculate_fid, getImages
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.losses import MSE
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""
This file generates the adversarial images for CIFAR-10 dataset 
using the Blackbox attack described in: https://arxiv.org/pdf/2001.11137.pdf
"""


train_img_path = "./trainCifar/train"
test_img_path = "./trainCifar/test"

MODEL_PATH = "./cifar_64.h5"

def loadModel():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def getData():
    origin_train_pair_list = []
    origin_test_pair_list = []

    # for i in range(x_train.shape[0]):
    #     pair = []
    #     pair.append(x[i])
    #     pair.append(y[i])
    #     origin_train_pair_list.append(pair)

    folder_name_train = os.listdir(train_img_path)
    folder_name_test = os.listdir(test_img_path)

    NUM_CLASS = len(folder_name_train)

    folder_path_train = []
    folder_path_test = []
    for i in range(NUM_CLASS):
        class_name = folder_name_train[i]
        path = train_img_path 
        path = path + '/' + class_name
        folder_path_train.append(path)

    for i in range(NUM_CLASS):
        class_name = folder_name_test[i]
        path = test_img_path
        path = path + '/' + class_name
        folder_path_test.append(path)
        
    image_label_pair_list_train = []

    #This will decide how many extra images you want to add to the dataset
    NUM_EXTRA_IMAGES = 5000

    #The initial label is 0
    label = 0

    for i in range(NUM_CLASS):
        folder_name = folder_path_train[i]
        image_name_list = os.listdir(folder_name)
        for ids in range(NUM_EXTRA_IMAGES):
            pair = []
            file_path = folder_name + '/' + image_name_list[ids]
            img = np.array(Image.open(file_path))
            pair.append(img)
            lab_list = []
            lab_list.append(label)
            pair.append(lab_list)
            image_label_pair_list_train.append(pair)
        label += 1

    label = 0
    image_label_pair_list_test = []
    for i in range(NUM_CLASS):
        folder_name = folder_path_test[i]
        image_name_list = os.listdir(folder_name)
        for ids in range(1000):
            pair = []
            file_path = folder_name + '/' + image_name_list[ids]
            img = np.array(Image.open(file_path))
            pair.append(img)
            lab_list = []
            lab_list.append(label)
            pair.append(lab_list)
            image_label_pair_list_test.append(pair)
        label += 1
        
    # We shuffle it again
    random.shuffle(image_label_pair_list_train)
    random.shuffle(image_label_pair_list_test)

    origin_train_pair_list.extend(image_label_pair_list_train)
    origin_test_pair_list.extend(image_label_pair_list_test)
    random.shuffle(origin_train_pair_list)
    random.shuffle(origin_test_pair_list)

    image_list_train = []
    label_list_train = []
    image_list_test = []
    label_list_test = []

    for i in range(len(origin_train_pair_list)):
        pair_train = origin_train_pair_list[i]
        image_list_train.append(pair_train[0])
        label_list_train.append(pair_train[1])

    for i in range(len(origin_test_pair_list)):
        pair_test = origin_test_pair_list[i]
        image_list_test.append(pair_test[0])
        label_list_test.append(pair_test[1])


    x_train = np.array(image_list_train)
    y_train = np.array(label_list_train)

    x_test = np.array(image_list_test)
    y_test = np.array(label_list_test)


    # x_train = x_train.reshape(x_train.shape[0], -1)
    # x_test = x_test.reshape(x_test.shape[0], -1)

    return x_train, y_train, x_test, y_test

def getTest():
    model = loadModel()
    folder_name = "../../OriginalImages_cifar64_XAI"
    image_name_list = os.listdir(folder_name)
    testX = []
    testY = []
    for image_name in image_name_list:
        image_path = folder_name + "/" + image_name
        img = np.array(Image.open(image_path))
        a = img.reshape((1, -1))/255
        output = model.predict(a)[0]
        label = np.argmax(output)
        testX.append(img)
        testY.append([label])


    testX = np.array(testX)
    testY = np.array(testY)
    testX = testX / 255.0
    return testX, testY

def getModel():
    print("[INFO] loading CIFAR dataset...")
    trainX, trainY, testX, testY = getData()
    trainX = trainX / 255.0
    testX = testX / 255.0
    trainX = np.expand_dims(trainX, axis=-1)
    testX = np.expand_dims(testX, axis=-1)
    trainY = to_categorical(trainY, 10)
    testY = to_categorical(testY, 10)
    model = tf.keras.models.load_model('../Models/cifar64Model.h5')
    testX, testY = getTest()
    testX = np.expand_dims(testX, axis=-1)
    testY = to_categorical(testY, 10)
    return model, testX, testY

def generate_image_adversary(image):
#   print(np.shape(image))
  image_array = np.array(image).reshape(1, 64*64*3)
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
  image = image_array.reshape(1, 64, 64, 3)
  return image

def convertToMtarix(array, m, n, channels):
    for i in range(m*n*channels):
        array[i] = 255*array[i]
    matrix = np.array(array)
    return matrix.reshape((m,n, channels))

def show(pixelMatrix, w, h, channels):
    # print(np.shape(pixelMatrix))
    pixelMatrix = convertToMtarix(pixelMatrix[0], w, h, channels)
    data = np.array(pixelMatrix)
    im = Image.fromarray(data.astype(np.uint8), mode='RGB')
    # im.show()
    return im

def generateAdversarial(model, testX, testY, folderSuffix):
    countOriginal = [0]*10
    countAdversary = [0]*10
    avConf = 0
    totalL2 = 0
    totalLinf = 0
    avCount = 0
    success = 0
    parentDir = "../Images/"
    # folderSuffix = ""
    
    for i in range(371):
        image = testX[i]
        original_array = np.array(image).reshape(1, 64*64*3)
        label = testY[i]
        adversary = generate_image_adversary(image.reshape(1, 64, 64, 3))
        pred = model.predict(adversary)
        adversary = adversary.reshape(64, 64, 3)
        adversarial_array = np.array(adversary).reshape(1, 64*64*3)
        
        l2 = np.linalg.norm(np.array(adversarial_array)-np.array(original_array))
        diff = np.array(adversarial_array[0])-np.array(original_array[0])
        diff = [abs(x) for x in diff]

        countChange = 0
        for x in diff:
            if x!=0:
                countChange = countChange + 1

        linf = np.argmax(diff)
        originalImage = show(original_array, 64, 64, 3)
        
        adversarialImage = show(adversarial_array, 64, 64, 3)
        originalLabel = np.argmax(label)
        adversaryLabel = np.argmax(pred)
        if originalLabel!=adversaryLabel:
            success = success + 1
        else:
            continue
        if countOriginal[originalLabel]<30:
            countAdversary[adversaryLabel] = 1 + countAdversary[adversaryLabel]
            countOriginal[originalLabel] = 1 + countOriginal[originalLabel]
        
        originalImage.save(parentDir+"OriginalImages"+folderSuffix+"/Image_"+str(i)+".jpg")
        adversarialImage.save(parentDir+"AdversarialImages"+folderSuffix+"/Image_"+str(i)+".jpg")

        print("Original Label =", np.argmax(label), "\nAdversarial Label=", np.argmax(pred), "\nL2-distance=", l2, "\nL-infinity-distance=", diff[int(linf)], "\nNumber of pixels changed=", countChange)
        totalL2 = totalL2 + l2
        totalLinf = totalLinf + diff[int(linf)]
        avCount = avCount + countChange
        print(pred[0][adversaryLabel])
        avConf = avConf + pred[0][adversaryLabel]
        print("..........................................")
    return avConf, totalL2, totalLinf, avCount, success, countOriginal, countAdversary

def displayResults():
    folderSuffix = "_same"

    model, textX, textY = getModel()
    t1 = time()
    avConf, totalL2, totalLinf, avCount, success, countOriginal, countAdversary = generateAdversarial(model, textX, textY, folderSuffix)
    t2 = time()
    print("Average Confidence:",avConf/success)
    print("Average L2:",totalL2/success)
    print("Average L-inf:",totalLinf/success)
    print("Average number of pixels modified:", avCount/success)
    print("Percentage of pixels modified(average):", (avCount/success)*100/(64*64*3)," %")
    print("Attack was successful was on:", success, "Images.")
    print("Average time taken: ", (t2-t1)/success, "seconds.")
    print(countOriginal)
    print(countAdversary)

    im1, im2 = getImages(folderSuffix)
    
    fid = calculate_fid(model, im1, im2)
    print(f'FID for {len(im1)} images is: {fid:.6f}')
    
    toCheck = countAdversary                                                                                                                                                        

    print("Pielou Meaure is: ",PielouMeaure(toCheck, len(toCheck)))
    print()

displayResults()
# getTest()
# getModel()
