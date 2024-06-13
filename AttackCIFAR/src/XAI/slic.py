import matplotlib.pyplot as plt
from itertools import product
from skimage.segmentation import slic, mark_boundaries
from skimage import io
import numpy as np
from csv import reader


def getData():
    inputs = []
    outputs = []
    f1 = open('../../data/MNIST/inputs.csv', 'r')
    f1_reader = reader(f1)
    stopAt = 500
    f2 = open('../../data/MNIST/outputs.csv', 'r')
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

class Bundle:
    def __init__(self):
        self.image = None
        self.n_segments = 50
        self.comp = 0.01 # lower the compactness, more closer is it to the shape
        self.segments = None

    def plot_from_lists(pixel_lists):
        # Create a 28x28 plot
        plot_array = np.zeros((28, 28))

        for pixel_value in pixel_lists:
            row = pixel_value // 28
            col = pixel_value % 28
            plot_array[row, col] = 255  # Assign a unique value for each list

        # Display the plot
        plt.imshow(plot_array, cmap='gray')
        plt.title("Mapped Plot from Lists")
        plt.show()

    def generate_segments(self, image_file, num_segs, comp, channel_axis=None):
        # Load an example image
        self.image = io.imread(image_file)
        self.num_segs = num_segs
        self.comp = comp

        # Perform superpixel segmentation using SLIC
        segments = slic(self.image, n_segments=self.num_segs, compactness=self.comp, channel_axis=channel_axis)
        self.segments = segments
        # self.draw_segments()

        # np.set_printoptions(threshold=np.inf, precision=2, suppress=True)
        # print(segments)

        # Get image dimensions
        if(channel_axis == None):
            height, width = self.image.shape
        else:
            height, width,_ = self.image.shape

        # Create a dictionary to store pixel positions for each segment
        segment_positions = {}

        # Iterate through each pixel and its segment label
        for i, j in product(range(height), range(width)):
            orig_img_pixel_val = np.float32(self.image[i,j] / 255.0)
            label = segments[i, j]
            if label not in segment_positions:
                segment_positions[label] = []
            segment_positions[label].append(((0, i * width + j), orig_img_pixel_val))


        # Convert the dictionary values to lists
        segment_positions_lists = list(segment_positions.values())

        return segment_positions_lists

    def generate_segments2(self, image_data, w, h, num_segs, comp, channel_axis=None):
        # Load an example image
        if channel_axis == None:
            self.image = image_data.reshape((w, h))
        else:
            self.image = image_data.reshape((w, h, len(image_data) // (w * h)))
        self.num_segs = num_segs
        self.comp = comp

        # Perform superpixel segmentation using SLIC
        segments = slic(self.image, n_segments=self.num_segs, compactness=self.comp, channel_axis=channel_axis)
        self.segments = segments
        # self.draw_segments()

        # np.set_printoptions(threshold=np.inf, precision=2, suppress=True)
        # print(segments)

        # Get image dimensions
        if(channel_axis == None):
            height, width = w, h
        else:
            height, width,_ = w, h, len(image_data) // (w * h)

        # Create a dictionary to store pixel positions for each segment
        segment_positions = {}

        # Iterate through each pixel and its segment label
        for i, j in product(range(height), range(width)):
            orig_img_pixel_val = np.float32(self.image[i,j] / 255.0)
            label = segments[i, j]
            if label not in segment_positions:
                segment_positions[label] = []
            segment_positions[label].append(((0, i * width + j), orig_img_pixel_val))


        # Convert the dictionary values to lists
        segment_positions_lists = list(segment_positions.values())

        return segment_positions_lists

    def draw_segments(self):
        # Display the original image with superpixel boundaries
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(self.image, cmap='gray')
        ax[0].set_title("Original Image")

        ax[1].imshow(mark_boundaries(self.image, self.segments))
        ax[1].set_title("Superpixel Segmentation")

        plt.show()


if __name__ == '__main__':
    B = Bundle()
    channel_axis = None
    segs = 50
    comp = 10
    # image_path = "/home/ritesh/Desktop/Images/digits/2_Digits_6.png"
    inp, out, _ = getData();
    # print(B.generate_segments(image_path, segs, comp, channel_axis))
    print(B.generate_segments2(np.array(inp[0]), 28, 28, segs, comp, channel_axis))
