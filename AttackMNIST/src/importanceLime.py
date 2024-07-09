import os
import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift, slic
import tensorflow as tf
import logging
import tqdm # type: ignore

tf.get_logger().setLevel(logging.ERROR)
# Disable tqdm globally
tqdm.tqdm().disable = True



# Define the segmentation function with adjusted parameters
def segmentation_fn(image):
    # return quickshift(image, kernel_size=1, max_dist=4, ratio=0.5)
    return slic(image, n_segments=50, compactness=10)


def loadImage(img_path, processed=True):
    # Load and preprocess your image as a grayscale image
    if not processed:
        img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image to [0, 1]
        img_array_reshaped = img_array.reshape(1, -1)[0]
    else:
        img_array = np.reshape(img_path, (1, 28, 28, 1))
        img_array_reshaped = img_path 

    # Convert grayscale image to RGB by repeating the single channel three times
    img_array_rgb = np.repeat(img_array, 3, axis=-1)

    return img_array_rgb, img_array_reshaped


def limeExplanation(model, img_path, MODEL_PATH=False, processed=True):

    if model == None:
        model = load_model(MODEL_PATH)

    # Define a function to predict with the model using a reshaped input
    def predict_fn(images):
        reshaped_images = images[:, :, :, 0].reshape(images.shape[0], -1)  # Use only one channel for prediction
        return model.predict(reshaped_images)

    img_array_rgb, img_array_reshaped = loadImage(img_path, processed)

    # Print the model prediction
    preds = predict_fn(img_array_rgb)
    # print(f"Predicted class: {np.argmax(preds)}")

    # Create a LIME image explainer object
    explainer = lime_image.LimeImageExplainer()

    with tqdm.tqdm(disable=True):
    # Generate an explanation for the instance
        explanation = explainer.explain_instance(img_array_rgb[0].astype('double'), predict_fn, top_labels=5, hide_color=0, num_samples=100, segmentation_fn=segmentation_fn)

    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]
    # print("Top label: ", ind)

    # Map each explanation weight to the corresponding superpixel
    dict_map = dict(explanation.local_exp[ind])
    # heatmap = np.vectorize(dict_map.get)(explanation.segments)

    sorted_dict_map = dict(sorted(dict_map.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_dict_map)
    # print(explanation.segments)
    segmented_img = explanation.segments.reshape(1, -1)[0]
    important_segments = {seg: [] for seg in sorted_dict_map.keys()}
    # print(important_segments)
    for i in range(784):
        important_segments[segmented_img[i]-1].append(((0, i), img_array_reshaped[i]))

    return list(important_segments.values())

if __name__ == '__main__':
    importance_bundle = limeExplanation(model=None, img_path='../Images/OriginalImages_max/Image_0.jpg', MODEL_PATH='../Models/MNIST/mnist.h5', processed=False)
    print(importance_bundle)
