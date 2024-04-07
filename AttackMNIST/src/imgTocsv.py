import os
import numpy as np
from csv import reader
from PIL import Image

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

# image_dir = '../Images/OriginalImages_max_XAI'
# output_file = "orig_data.csv"
# out_class = "orig_y.csv"

image_dir = '../Images/AdversarialImages_max_XAI'
output_file = "adv_data.csv"
out_class = "adv_y.csv"

MODEL_PATH = '../Models/mnist.h5'
model = tf.keras.models.load_model(MODEL_PATH)

def predict(model, inp):
    logits = model.predict([inp])
    y_test = np.zeros(10)
    y_idx = np.argmax(logits)
    y_test[y_idx] = 1
    print(y_test)
    return y_test


def getData(data):
    inputs = []
    f1 = open(data, 'r')
    f1_reader = reader(f1)
    for row in f1_reader:
        inp = [float(x) for x in row]
        inputs.append(inp)
    return inputs

def images_to_csv(image_paths, output_file):
    with open(output_file, 'w') as f:
        for image_path in image_paths:
            # Load the image
            img = Image.open(image_path)
            
            # Convert the image to grayscale if needed
            img = img.convert('L')  # Convert to grayscale
            
            # Resize the image to 28x28 if needed
            img = img.resize((28, 28))

            # Convert the image to a numpy array
            img_array = np.array(img) / 255.0

            # Flatten the image into a 1D array
            img_flat = img_array.flatten()

            # Write the flattened image data to the CSV file
            f.write(','.join(map(str, img_flat)) + '\n')# Directory containing the images

def img_to_class(output_file, out_class):
    inputs = getData(output_file)
    with open(out_class, 'w') as f2:
        for inp in inputs:
            y_test = predict(model, inp)
            f2.write(",".join(map(str, y_test)) + "\n")

# Get a list of all files in the directory
all_files = os.listdir(image_dir)

# Filter out only the image files (assuming images have extensions like '.png', '.jpg', etc.)
image_files = [f for f in all_files if f.lower().endswith(('.jpg'))]

image_sorted = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))
# Construct the full paths to the image files
image_paths = [os.path.join(image_dir, f) for f in image_sorted]

# Call the function to convert images to CSV
images_to_csv(image_paths, output_file)
img_to_class(output_file, out_class)
