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
adv_path = ['Image_41.jpg', 'Image_10.jpg', 'Image_95.jpg', 'Image_335.jpg']
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
