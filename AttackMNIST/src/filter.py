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
# input_csv = '../data/FMNIST/inputs.csv'
input_csv = '../data/FMNIST/outputs.csv'

# Specify the output CSV file path
# output_csv = 'failedAttack_fmnist2_inputs.csv'
output_csv = 'failedAttack_fmnist2_outputs.csv'

# List of row indices to keep
# FMNIST 1
# rows_to_keep = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 122, 123, 124, 125, 126, 127, 128, 129, 130, 134, 135, 136, 137, 139, 140, 141, 142, 143, 144, 146, 147, 148, 149, 150, 151, 153, 154, 156, 157, 159, 160, 161, 162, 163, 164, 167, 168, 169, 170, 171, 172, 174, 175, 176, 178, 179, 180, 181, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 225, 226, 227, 228, 231, 232, 233, 234, 235, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251, 252, 253, 254, 255, 256, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 271, 272, 273, 275, 276, 277, 278, 281, 282, 283, 284, 285, 286, 287, 289, 290, 291, 292, 293, 295, 297, 298, 299, 301, 302, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 340, 342, 344, 345, 347, 350, 351, 352, 353, 354, 355, 356, 358, 359, 360, 361, 362, 365, 368, 370, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 383, 385, 386, 387, 388, 390, 391, 392, 394, 395, 397, 399, 400, 401, 402, 403, 404, 405, 407, 408, 410, 412, 413, 414, 415, 416, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 429, 430, 431, 432, 433, 434, 438, 439, 440, 441, 442, 443, 444, 445, 446, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 460, 461, 462, 464, 465, 468, 470, 471, 473, 474, 476, 477, 480, 481, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 513, 514, 515, 516, 517, 519, 520, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 548, 549, 550, 551, 553, 555, 556, 557, 559, 560, 562, 563, 564, 565, 566, 567, 568, 570, 571, 572, 573, 574, 575, 578, 579, 580, 581, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 606, 607, 608, 609, 610, 611, 612, 613, 615, 616, 617, 622, 623, 624, 625, 626, 627, 628, 630, 631, 633, 634, 635, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 656, 658, 659, 660, 661, 662, 663, 664, 665, 669, 670, 671, 672, 673, 675, 676, 677, 678, 680, 681, 682, 684, 685, 686, 687, 688, 692, 693, 695, 696, 698, 699, 700, 701, 703, 704, 707, 708, 709, 710, 711, 712, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 730, 731, 732, 733, 735, 736, 737, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 758, 759, 761, 762, 763, 765, 766, 767, 768, 769, 771, 772, 773, 776, 778, 779, 780, 781, 782, 783, 785, 786, 787, 788, 789, 791, 792, 793, 794, 795, 796, 797, 799, 800, 801, 802, 803, 805, 806, 807, 808, 809, 810, 811, 812, 813, 815, 816, 817, 818, 819, 820, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 835, 836, 838, 839, 840, 841, 842, 844, 846, 849, 851, 852, 853, 854, 855, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 888, 889, 891, 892, 893, 894, 895, 896, 897, 900, 901, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 917, 918, 922, 923, 924, 925, 929, 930, 931, 932, 933, 934, 935, 937, 938, 939, 940, 941, 942, 944, 945, 946, 948, 949, 950, 951, 954, 955, 957, 958, 959, 960, 961, 962, 963, 965, 966, 967, 968, 969, 970, 971, 972, 973, 975, 977, 979, 980, 981, 982, 984, 985, 987, 988, 989, 990, 991, 992, 993, 996, 997, 998, 999] 

# FMNIST 2
rows_to_keep = [0, 1, 5, 7, 9, 10, 11, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 35, 36, 37, 38, 39, 40, 42, 44, 45, 48, 49, 53, 55, 56, 57, 59, 60, 61, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 79, 80, 84, 86, 88, 97, 98, 99, 100, 101, 102, 103, 105, 106, 107, 109, 110, 112, 114, 116, 117, 121, 122, 124, 125, 128, 130, 135, 139, 140, 143, 144, 146, 147, 149, 150, 151, 154, 156, 157, 161, 163, 165, 168, 169, 170, 171, 172, 173, 174, 176, 178, 179, 180, 181, 185, 188, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 203, 205, 206, 208, 209, 212, 213, 214, 215, 216, 218, 219, 220, 222, 225, 226, 228, 231, 232, 233, 237, 238, 239, 240, 241, 242, 243, 245, 246, 253, 254, 255, 258, 263, 264, 265, 266, 267, 268, 269, 271, 272, 273, 276, 278, 281, 284, 285, 287, 289, 290, 291, 292, 295, 297, 298, 299, 301, 302, 304, 306, 308, 309, 311, 312, 313, 314, 315, 316, 317, 318, 322, 324, 326, 328, 329, 330, 331, 333, 335, 336, 337, 338, 350, 352, 356, 358, 359, 361, 362, 363, 365, 368, 371, 372, 376, 380, 381, 383, 385, 387, 388, 390, 391, 394, 395, 397, 399, 410, 412, 414, 415, 416, 418, 419, 421, 422, 424, 426, 427, 430, 431, 441, 442, 443, 444, 446, 448, 450, 451, 453, 454, 457, 458, 459, 461, 464, 465, 468, 473, 474, 476, 477, 480, 481, 483, 486, 488, 490, 491, 492, 493, 494, 497, 498, 500, 502, 503, 504, 506, 507, 508, 510, 511, 513, 514, 517, 519, 520, 523, 524, 525, 526, 528, 530, 531, 535, 538, 539, 540, 541, 543, 545, 549, 550, 553, 555, 556, 557, 559, 560, 562, 563, 564, 565, 566, 567, 568, 570, 571, 572, 578, 579, 580, 583, 584, 586, 587, 588, 589, 590, 592, 595, 599, 601, 603, 605, 606, 608, 609, 610, 612, 613, 615, 616, 617, 622, 624, 625, 628, 630, 631, 633, 634, 635, 637, 639, 640, 641, 642, 643, 645, 646, 647, 648, 649, 650, 652, 654, 658, 659, 660, 661, 662, 663, 664, 665, 669, 671, 672, 673, 675, 676, 677, 678, 680, 682, 684, 692, 693, 695, 696, 703, 704, 707, 708, 709, 711, 712, 715, 716, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 732, 733, 735, 736, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 752, 753, 754, 758, 759, 761, 763, 765, 766, 768, 769, 771, 772, 773, 774, 777, 778, 780, 786, 791, 793, 795, 796, 797, 799, 800, 803, 805, 806, 810, 811, 812, 815, 816, 818, 819, 822, 823, 825, 828, 829, 838, 839, 842, 844, 846, 851, 853, 859, 861, 863, 866, 867, 872, 873, 877, 878, 879, 880, 881, 882, 883, 885, 886, 888, 889, 891, 893, 894, 895, 896, 897, 899, 900, 901, 902, 905, 906, 907, 909, 910, 911, 912, 914, 918, 919, 922, 923, 925, 927, 929, 930, 931, 932, 933, 934, 935, 938, 939, 940, 941, 942, 944, 946, 947, 949, 951, 954, 955, 958, 959, 960, 962, 965, 966, 967, 968, 970, 972, 977, 979, 980, 981, 982, 984, 985, 987, 990, 991, 992, 996, 998, 999]

# Call the function to filter rows and save to a new CSV file
filter_csv_rows(input_csv, output_csv, rows_to_keep)
