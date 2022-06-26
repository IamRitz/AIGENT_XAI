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

def getModel():
    print("[INFO] loading MNIST dataset...")
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainX = trainX / 255.0
    testX = testX / 255.0
    trainX = np.expand_dims(trainX, axis=-1)
    testX = np.expand_dims(testX, axis=-1)
    trainY = to_categorical(trainY, 10)
    testY = to_categorical(testY, 10)
    model = tf.keras.models.load_model('cifarModel.h5')
    return model, testX, testY

def generate_image_adversary(model, image, label, eps=2 / 255.0):
  image = tf.cast(image, tf.float32)
  with tf.GradientTape() as tape:
    tape.watch(image)
    pred = model(image)
    loss = MSE(label, pred)
    gradient = tape.gradient(loss, image)
    signedGrad = tf.sign(gradient)
    adversary = (image+ (signedGrad*eps)).numpy()
    return adversary

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
        original_array = np.array(image).reshape(1, 32*32*3)
        label = testY[i]
        adversary = generate_image_adversary(model, image.reshape(1, 32, 32, 3), label, eps=0.1)
        pred = model.predict(adversary)
        adversary = adversary.reshape(32, 32, 3)
        adversarial_array = np.array(adversary).reshape(1, 32*32*3)
        
        l2 = np.linalg.norm(np.array(adversarial_array)-np.array(original_array))
        diff = np.array(adversarial_array[0])-np.array(original_array[0])
        diff = [abs(x) for x in diff]

        countChange = 0
        for x in diff:
            if x!=0:
                countChange = countChange + 1

        linf = np.argmax(diff)
        originalImage = show(original_array, 32, 32, 3)
        
        adversarialImage = show(adversarial_array, 32, 32, 3)
        originalLabel = np.argmax(label)
        adversaryLabel = np.argmax(pred)
        if originalLabel!=adversaryLabel:
            success = success + 1
        else:
            continue
        if countOriginal[originalLabel]<30:
            countAdversary[adversaryLabel] = 1 + countAdversary[adversaryLabel]
            countOriginal[originalLabel] = 1 + countOriginal[originalLabel]
        
        originalImage.save("OriginalImages1/Image_"+str(i)+".jpg")
        adversarialImage.save("AdversarialImages1/Image_"+str(i)+".jpg")

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
    folderCount = 1

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

displayResults()