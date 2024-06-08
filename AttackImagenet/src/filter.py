import pandas as pd

# Function to filter rows from a CSV file
def filter_csv_rows(input_csv, output_csv, rows_to_keep):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv, header=None)
    
    # Filter the DataFrame to keep only the specified rows
    filtered_df = df.iloc[rows_to_keep]
    
    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv, index=False, header=False)

# Specify the input CSV file path
input_csv = '../data/IMAGENET5/inputs.csv'
# input_csv = '../data/IMAGENET5/outputs.csv'

# Specify the output CSV file path
# output_csv = 'fadv_img1_inputs.csv'
# output_csv = 'fadv_img1_outputs.csv'

# output_csv = 'fadv_img2_inputs.csv'
# output_csv = 'fadv_img2_outputs.csv'

# List of row indices to keep

#imgnet-1
# rows_to_keep = [1, 2, 5, 8, 12, 27, 33, 41, 51, 71, 90, 91, 94, 95, 104, 107, 109, 110, 112, 115, 118, 119, 121, 122, 126, 134, 137, 138, 140, 141, 144, 160, 166, 168, 180, 184, 189, 191, 198, 199, 206, 210, 218, 222, 226, 233, 237, 244, 252, 255, 257, 260, 267, 268, 287, 300, 302, 319, 342, 343, 361, 362, 367, 370, 372, 375, 380, 383, 394, 403, 406, 408, 413, 416, 417, 418, 421, 427, 430, 433, 435, 445, 447, 454, 455, 460, 462, 471, 479, 486, 487, 490, 494, 501, 503, 504, 526, 532, 534, 542, 546, 553, 555, 558, 560, 567, 573, 578, 580, 589, 590]

# imgnet-2
rows_to_keep = [0, 11, 12, 18, 21, 23, 32, 35, 40, 42, 46, 47, 54, 56, 65, 70, 76, 81, 86, 93, 98, 100, 110, 111, 116, 117, 122, 124, 127, 128, 149, 150, 152, 155, 159, 161, 164, 165, 169, 172, 174, 177, 190, 195, 200, 201, 202, 213, 217, 219, 223, 227, 234, 236, 239, 240, 249, 253, 264, 269, 270, 272, 274, 277, 280, 288, 289, 290, 291, 297, 311, 314, 318, 321, 322, 323, 326, 335, 336, 338, 341, 345, 347, 351, 356, 358, 360, 363, 371, 377, 381, 385, 386, 387, 391, 398, 401, 404, 405, 409, 412, 414, 421, 425, 428, 429, 432, 434, 435, 440, 441, 443, 453, 456, 457, 458, 464, 470, 472, 474, 475, 477, 489, 502, 505, 508, 511, 514, 520, 521, 525, 528, 539, 541, 544, 545, 557, 561, 564, 575, 577, 583, 584, 591, 599]

# Call the function to filter rows and save to a new CSV file
filter_csv_rows(input_csv, output_csv, rows_to_keep)

