# code for displaying multiple images in one figure

#import libraries
import cv2
from matplotlib import pyplot as plt
import os

# create figure
x, y = 4, 2
fig = plt.figure(figsize=(x, y))

rows = 2
columns = 4

originalImages = []
adversarialImages = []

for path in os.listdir("AdversarialImages"):
    originalImages.append('OriginalImages/'+str(path))
    adversarialImages.append('AdversarialImages/'+str(path))

counterOrig = 0
counterAdver = 0
f=0
t=0
adv_path = ['Image_34.jpg', 'Image_324.jpg', 'Image_422.jpg', 'Image_432.jpg']
for i in range(len(adv_path)):
    t = t+1
    fig.add_subplot(rows, columns, t)
    file = cv2.imread("OriginalImages/"+adv_path[i])
    plt.imshow(file)
    plt.axis('off')

for i in range(len(adv_path)):
    t = t+1
    fig.add_subplot(rows, columns, t)
    file = cv2.imread("AdversarialImages/"+adv_path[i])
    plt.imshow(file)
    plt.axis('off')
plt.savefig("Grid_"+str(f)+".jpg")
# while(counterAdver<len(adversarialImages)):
#     t = 1
#     # flag = -1
#     print("...........................................")
#     print("For:", f)
#     for i in range(columns):
#         if counterOrig>=len(adversarialImages):
#             # flag = 1
#             break
#         fig.add_subplot(rows, columns, t)
#         t = t + 1
#         print(originalImages[counterOrig])
#         file = cv2.imread(originalImages[counterOrig])
#         plt.imshow(file)
#         counterOrig = counterOrig + 1
#         plt.axis('off')

#     for i in range(columns):
#         if counterAdver>=len(adversarialImages):
#             # flag =1
#             break
#         fig.add_subplot(rows, columns, t)
#         t = t + 1
#         print(adversarialImages[counterAdver])
#         file = cv2.imread(adversarialImages[counterAdver])
#         plt.imshow(file)
#         counterAdver = counterAdver + 1
#         plt.axis('off')
#     print("...........................................")
#     if len(adversarialImages)-counterAdver<=columns:
#         columns = len(adversarialImages)-counterAdver
#     plt.savefig("Grids/comparison_"+str(f)+".jpg")
#     f = f + 1
#     fig = plt.figure(figsize=(x, y))

