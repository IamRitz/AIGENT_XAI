from csv import reader
import csv
import os
from re import I
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
import numpy as np


f1 = open('MNISTdata/adversarialData-2.csv', 'r')
f1_reader = reader(f1)
count = 0 
model = tf.keras.models.load_model(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/Models/mnist.h5')
i = 0
for row in f1_reader:
    # count = count+1
    if i%500==0 and i!=0:
        print(count, i, (count/i*100),"%")
    i = i + 1
    length = len(row)
    output = float(row[length-1])
    inp = [float(x) for x in row[:length-1]]
    predicted_output = model.predict([inp])
    predicted_label = float(np.argmax(predicted_output))
    # print(type(predicted_label), type(output))
    if predicted_label != output:
        count = count + 1
    # else:
    #     print(inp, predicted_output, output)
    # break
print(count)