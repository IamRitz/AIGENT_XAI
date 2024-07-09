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
    return slic(image, n_segments=250, compactness=10)


def loadImage(img_path, processed=True):
    # Load and preprocess your image as a grayscale image
    if not processed:
        img = image.load_img(img_path, target_size=(64, 64), color_mode='rgb')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image to [0, 1]
        img_array_reshaped = img_array.reshape(1, -1)[0]
    else:
        img_array = np.reshape(img_path, (64, 64, 3))
        img_array = np.expand_dims(img_array, axis=0)
        img_array_reshaped = img_path 

    return img_array, img_array_reshaped


def limeExplanation(model, img_path, MODEL_PATH=False, processed=True):

    if model == None:
        model = load_model(MODEL_PATH)

    # Define a function to predict with the model using a reshaped input
    def predict_fn(images):
        print(images.shape)
        reshaped_images = images.reshape(images.shape[0], -1)
        return model.predict(reshaped_images)

    img_array_rgb, img_array_reshaped = loadImage(img_path, processed)

    # Print the model prediction
    preds = predict_fn(img_array_rgb)
    print(f"Predicted class: {np.argmax(preds)}")

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
    # print(img_array_reshaped.shape)
    segmented_img = explanation.segments.reshape(1, -1)[0]
    important_segments = {seg: [] for seg in sorted_dict_map.keys()}
    # print(segmented_img)
    # print(important_segments)

    for i in range(64*64):
        important_segments[segmented_img[i]-1].append(((0, i), img_array_reshaped[i]))
        important_segments[segmented_img[i]-1].append(((0, i+64*64), img_array_reshaped[i+64*64]))
        important_segments[segmented_img[i]-1].append(((0, i+2*64*64), img_array_reshaped[i+2*64*64]))

    return list(important_segments.values())

if __name__ == '__main__':
    importance_bundle = limeExplanation(model=None, img_path='../Images/OriginalImages/Image_0.jpg', MODEL_PATH='../Models/CIFAR10/cifar.h5', processed=False)
    print(importance_bundle)
