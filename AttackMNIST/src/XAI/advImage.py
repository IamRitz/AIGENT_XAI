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
image_path = '../Images/digits/0_Digits_1001.png'
mnist_image = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale

# Normalize the pixel values to be in the range [0, 1]
mnist_image = mnist_image / 255.0

# Display the original MNIST digit image
plt.imshow(mnist_image, cmap='gray')
plt.title('Original MNIST Digit Image')
plt.show()

# Define the minimal explanation list [((input_name, pixel_index), pixel_value), ...]
minimal_explanation = [((0, 10), 0.5), ((0, 20), 0.8)]

# Initialize the adversarial image
input_name = minimal_explanation[0][0][0]
input_shape = ort_session.get_inputs()[0].shape
# print(input_shape)
adversarial_image = np.expand_dims(mnist_image, axis=0)  # Add batch dimension

# Perturb the minimal explanation pixels
for (input_name, pixel_index), pixel_value in minimal_explanation:
    pixel_index = int(pixel_index)
    adversarial_image[0, 0, pixel_index // input_shape[2], pixel_index % input_shape[3]] = pixel_value

# Perturb the original image by a small factor
epsilon = 0.01
perturbed_image = mnist_image + epsilon * adversarial_image[0, 0]

# Clip the values to be within the valid pixel range [0, 1]
perturbed_image = np.clip(perturbed_image, 0, 1)

# Run the ONNX model to get predictions for the perturbed image
perturbed_predictions = ort_session.run(None, {input_name: np.expand_dims(perturbed_image, axis=0)})

# Display the perturbed image and observe the ONNX predictions
plt.imshow(perturbed_image, cmap='gray')
plt.title('Perturbed Image\nPredictions: {}'.format(perturbed_predictions))
plt.show()
