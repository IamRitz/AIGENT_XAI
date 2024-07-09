import os
from csv import reader
import numpy as np
from sklearn.calibration import calibration_curve
import scipy.special
from scipy import stats
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

# MODEL_PATH = '../../Models/cifar.h5'
# MODEL_PATH = '../../Models/cifar1.h5'
MODEL_PATH = '../../Models/cifar2.h5'

diff_arr = []
lar_arr = []

count = 0
def loadModel():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def getData(inp_path, out_path):
    inputs = []
    f1 = open(inp_path, 'r')
    f1_reader = reader(f1)

    outputs = []
    f2 = open(out_path, 'r')
    f2_reader = reader(f2)
    for row in f1_reader:
        inp = [float(x) for x in row]
        inputs.append(inp)
    for row in f2_reader:
        out = [float(x) for x in row]
        outputs.append(out)
    return inputs, outputs, len(inputs)

def confidence_score(model, X_test, y_test, i):
    global diff_arr
    global lar_arr
    global count
    logits = model.predict(np.array([X_test]))[0]
    probabilities = scipy.special.softmax(logits)
    largest = np.partition(probabilities, -1)[-1]
    second_largest = np.partition(probabilities, -2)[-2]
    # print(probabilities)
    # print(f"Largest: {largest}")
    # print(f"Second Largest: {second_largest}")
    # print(f"Diff: {largest-second_largest}")
    diff = largest - second_largest
    lar_arr.append(round(largest, 3)) 
    diff_arr.append(round(diff, 3))
    if diff >= 0.5:
        count += 1

def confidence_score2(model, X_test, y_test, i):
    global count
    global orig_low_sc

    logits = model.predict(np.array([X_test]))[0]
    probabilities = scipy.special.softmax(logits)
    # print(probabilities)

    true_labels = y_test

    # Compute the reliability curve (calibration curve)
    prob_true, prob_pred = calibration_curve(true_labels, probabilities[:], n_bins=10)

    # Platt scaling calibration
    calibrated_probabilities = np.interp(probabilities[:], prob_pred, prob_true)

    # print(calibrated_probabilities)
    # Calibrated confidence score (maximum probability)
    calibrated_confidence_score = calibrated_probabilities.max()

    if calibrated_confidence_score < 1:
        # print(probabilities)
        # print(calibrated_probabilities)
        out_class = np.argmax(probabilities) 
        print("Image: ", i)
        print("Predicted: ", out_class)
        out_class = np.argmax(y_test)
        print("True: ", out_class)
        print("Prob: ", probabilities[out_class])
        print("Calibrated Confidence Scores:", calibrated_confidence_score)
        count += 1


if __name__ == "__main__":
    inp_path = "../../data/inputs.csv"
    out_path = "../../data/outputs.csv"
    model = loadModel()
    inputs, outputs, _ = getData(inp_path, out_path)
    # print(inputs[0])
    for i in range(len(inputs)):
        print(f"------------------{i+1}----------------------")
        confidence_score(model, inputs[i], outputs[i], i+1)

    
    print("(Mean):",np.mean(lar_arr))
    print("(Median):",np.median(lar_arr))
    mode = stats.mode(lar_arr)
    print("(Mode value):", mode[0])
    print("(Mode count):", mode[1])
    print("Largest Confidence score: ", np.max(lar_arr))
