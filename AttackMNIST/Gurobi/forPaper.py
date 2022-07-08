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

for path in os.listdir("AdversarialImages2"):
    originalImages.append('OriginalImages2/'+str(path))
    adversarialImages.append('AdversarialImages2/'+str(path))

counterOrig = 0
counterAdver = 0
f=0
t=0
adv_path = ['Image_61.jpg', 'Image_41.jpg', 'Image_6.jpg', 'Image_33.jpg']
for i in range(len(adv_path)):
    t = t+1
    fig.add_subplot(rows, columns, t)
    file = cv2.imread("OriginalImages2/"+adv_path[i])
    plt.imshow(file)
    plt.axis('off')

for i in range(len(adv_path)):
    t = t+1
    fig.add_subplot(rows, columns, t)
    file = cv2.imread("AdversarialImages2/"+adv_path[i])
    plt.imshow(file)
    plt.axis('off')
plt.savefig("Grid_"+str(f)+".jpg")
