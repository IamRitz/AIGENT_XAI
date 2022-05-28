import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
"""
To supress the tensorflow warnings. 
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
from csv import reader
import csv
from fileinput import filename
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
"""
Setting verbosity of tensorflow to minimum.
"""
from time import time
from adversarialExampleMNIST import find

"""
This file generates adversarial examples for all inputs in the dataset of MNIST.
The adversarial images are generated with a specific limit of l-inf and total change required.
This file uses the functions defined in Gurobi/adversarialExampleMNIST.py, particularly the function find.
"""
def getData():
    inputs = []
    outputs = []
    f1 = open('MNISTdata/inputs.csv', 'r')
    f1_reader = reader(f1)

    f2 = open('MNISTdata/outputs.csv', 'r')
    f2_reader = reader(f2)

    for row in f1_reader:
        inp = [float(x) for x in row]
        inputs.append(inp)

    for row in f2_reader:
        out = [float(x) for x in row]
        outputs.append(out)

    return inputs, outputs, len(inputs)

def storeData(data, fileName, mode):
    f = open(fileName, mode)
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)
    f.close()
    return 0

def generateAattack():
    print("Reading data...")
    inputs, outputs, count = getData()
    print("########################\nData read. \nTotal number of samples in dataset: ", count,"\nBeginning attack.\n########################")
    status_codes = [0, 0, 0]
    """
    1 stands for feasible model, 2 stands for infeasible model, 0 stands for error occurred while model creation.
    """
    fileName = "MNISTdata/adversarialData-2.csv"
    model = tf.keras.models.load_model(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/Models/mnist.h5')
    t1 = time()
    adversarial_data = []
    anomaly = 0
    for i in range(count):
        inp = inputs[i]
        out = outputs[i]
        true_label = (np.argmax(out))
        num_outputs = len(out)
        num_inputs = len(inp)
        # print(inp, "\nOutputs:", out, "\nTrueLabel:", true_label, "\nNum out:", num_outputs, "\nNum In:", num_inputs)
        epsilons, status = find(0.2, 10, model, inp, true_label, num_inputs, num_outputs)
        status_codes[status] = status_codes[status] + 1
        if status==1:
            new_image = np.add(np.array(inp), np.array(epsilons))
            predicted_output = model.predict([new_image.tolist()])
            predicted_label = float(np.argmax(predicted_output))
            if predicted_label==true_label:
                # print("Found anomaly.", predicted_label, true_label)
                anomaly = anomaly + 1
                # break
            new_image = np.append(new_image,[true_label])
            adversarial_data.append(new_image)
        
        if i%100==0 and i!=0:
            t2 = time()
            print("\n####################################")
            print("Adversarial images generated for: ",(i+1)," images.")
            print("Status till now: ", status_codes) 
            print("Time taken till now: ",(t2 - t1)/3600, "hours.")
            print("Anomalies found: ", anomaly)
            mode = 'a'
            if i==100:
                mode = 'w'
            storeData(adversarial_data, fileName, mode)
            adversarial_data = []
            print("####################################")
    return adversarial_data

if __name__ == '__main__':
    adversarialData = generateAattack()
    print("########################\nAttack successfull. Adversarial examples writen to file for future use.\n########################")