from csv import reader
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = '../../Models/FMNIST/fashion_mnist1.h5'
# INP_DATA = '../../data/FMNIST/inputs.csv'
# OUT_DATA = '../../data/FMNIST/outputs.csv'
INP_DATA = '../failedAttack_fmnist1_inputs.csv'
OUT_DATA = '../failedAttack_fmnist1_outputs.csv'

def getData():
    inputs = []
    outputs = []
    f1 = open(INP_DATA, 'r')
    f1_reader = reader(f1)
    stopAt = 20
    f2 = open(OUT_DATA, 'r')
    f2_reader = reader(f2)
    i=0
    for row in f1_reader:
        inp = [float(x) for x in row]
        inputs.append(inp)
        i=i+1
        if i==stopAt:
            break
    i=0
    for row in f2_reader:
        out = [float(x) for x in row]
        outputs.append(out)
        i=i+1
        if i==stopAt:
            break
    return inputs, outputs, len(inputs)

"""
# Load and preprocess your image (ensure the image is 128x128)
img_path = '/home/ritesh/Desktop/Images/fashion_renamed/fashion_2.png'  # Replace with your image path
img = image.load_img(img_path, color_mode = 'grayscale', target_size = (28,28))
img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize the image to [0, 1]
print(img_array.shape)
arr=img_array.reshape((1, -1))[0]
"""







def pgd_attack(model, image, label, epsilon, alpha, num_iterations, sparsity):
    adv_image = tf.identity(image)

    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(tf.reshape(adv_image, (-1, 28*28)))
            loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)

        gradient = tape.gradient(loss, adv_image)
        signed_grad = tf.sign(gradient)

        flat_grad = tf.reshape(signed_grad, [-1])
        _, indices = tf.nn.top_k(tf.abs(flat_grad), k=sparsity, sorted=False)
        mask = tf.scatter_nd(tf.expand_dims(indices, 1), tf.ones_like(indices, dtype=tf.float32), flat_grad.shape)
        mask = tf.reshape(mask, signed_grad.shape)


        adv_image = adv_image + alpha * signed_grad * mask
        perturbation = tf.clip_by_value(adv_image - image, -epsilon, epsilon)
        adv_image = tf.clip_by_value(image + perturbation, 0, 1)  # Ensure values are within [0, 1]

    return adv_image



def performAttack():
    inputs, outputs, counts = getData()

    # Load your pre-trained model
    model = load_model(MODEL_PATH)

    # Set parameters for the PGD attack
    epsilon = 0.05                  # Maximum perturbation
    alpha = 0.01                    # Step size
    num_iterations = 40            # Number of iterations
    sparsity = 120
    counts = 1
    for i in range(counts):
        arr = np.array(inputs[13])
        img_array = arr.reshape((28, 28, 1))


        # Perform the prediction
        # print(img_array)
        # print(img_array.reshape((1, -1)).shape)
        # print(len(img_array[0]))
        predictions = model.predict(arr.reshape(1, -1)) 
        # predictions = model.predict(img_array.reshape(1, -1)) 

        # print(predictions)
        print(np.argmax(predictions))
        softmax_predictions = tf.nn.softmax(predictions)
        print("Softmax: ", softmax_predictions)

        # Convert the image to a tensor
        input_image = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Define the true label for the image (use the correct label for your image)
        true_label = tf.convert_to_tensor([np.argmax(predictions)])  # Replace with the actual label of your image


        # Generate adversarial image
        adversarial_image = pgd_attack(model, input_image, true_label, epsilon, alpha, num_iterations, sparsity)


    # adversarial_image = adversarial_image.numpy().reshape((1, -1))
    # print(adversarial_image.shape)
    # print(len(adversarial_image))
    # print("Reshape: ", adversarial_image.numpy().reshape(1, -1).shape)
    # Perform the prediction

        new_img = adversarial_image.numpy().reshape(1, -1);
        predictions = model.predict(adversarial_image.numpy().reshape(1, -1))

        # print(predictions)
        print(np.argmax(predictions))
        softmax_predictions = tf.nn.softmax(predictions)
        print("Softmax: ", softmax_predictions)
        # print(list(new_img[0]))

        # print(list(new_img[0]))
        # count = 0
        # for i,j in zip(img_array.reshape((1, -1))[0], new_img[0]):
        #     print(abs(i-j))
        #     if abs(i-j) > 0:
        #         count += 1
        #
        # print(count)
        print(list(adversarial_image.numpy().reshape(1,-1)[0]))

performAttack()
