import onnx
import onnxruntime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the ONNX model
onnx_model_path = './networks/path_to_save_model.onnx'
onnx_model = onnx.load(onnx_model_path)

# Create an ONNX runtime session
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# Load an example MNIST digit image
image_path = '../Images/digits/4_Digits_10.png'
mnist_image = np.array(Image.open(image_path).convert('L')).reshape(784,)  # Flatten to shape [784,]

# Normalize the pixel values to be in the range [0, 1]
mnist_image = mnist_image / 255.0

# Display the original MNIST digit image
plt.imshow(mnist_image.reshape(28, 28), cmap='gray')
plt.title('Original MNIST Digit Image')
plt.show()

# Define the minimal explanation list [(pixel_index, pixel_value), ...]
minimal_explanation = [((0, 10), 0.1), ((0, 20), 0.4)]

# Initialize the adversarial image
adversarial_image = mnist_image.copy()

# Perturb the minimal explanation pixels
for pixel_pos, pixel_value in minimal_explanation:
    pixel_index = pixel_pos[1]
    adversarial_image[pixel_index] = pixel_value

# Perturb the original image by a small factor
epsilon = 0.01
perturbed_image = mnist_image + epsilon * adversarial_image

# Clip the values to be within the valid pixel range [0, 1]
perturbed_image = np.clip(perturbed_image, 0, 1)

# Get the input name for the ONNX model
input_name = ort_session.get_inputs()[0].name

# Explicitly cast the perturbed_image to float32
perturbed_image = perturbed_image.astype(np.float32)

# Run the ONNX model to get predictions for the perturbed image
perturbed_predictions = ort_session.run(None, {input_name: perturbed_image.reshape(1, 784)})

print(perturbed_predictions)

# Extract the class with the maximum probability
predicted_class = np.argmax(perturbed_predictions)

print(predicted_class)

# # Display the perturbed image and observe the ONNX predictions
plt.imshow(perturbed_image.reshape(28, 28), cmap='gray')
plt.title('Perturbed Image\nPredictions: {}'.format(perturbed_predictions))
plt.show()
