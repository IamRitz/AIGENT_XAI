# code for displaying multiple images in one figure

#import libraries
import cv2
from matplotlib import pyplot as plt
import os

# create figure
x, y = 5, 1.5
fig = plt.figure(figsize=(x, y))

# setting values to rows and column variables
rows = 2
columns = 10

originalImages = []
adversarialImages = []

for path in os.listdir("/home/tooba/Documents/DNNVerification/AttackMNIST/Gurobi/AdversarialImages"):
    originalImages.append('OriginalImages/'+str(path))
    adversarialImages.append('AdversarialImages/'+str(path))

counterOrig = 0
counterAdver = 0
f=0

while(counterAdver<len(adversarialImages)):
    t = 1
    # flag = -1
    for i in range(columns):
        if counterOrig>=len(adversarialImages):
            # flag = 1
            break
        fig.add_subplot(rows, columns, t)
        t = t + 1
        file = cv2.imread(originalImages[counterOrig])
        plt.imshow(file)
        counterOrig = counterOrig + 1
        plt.axis('off')

    for i in range(columns):
        if counterAdver>=len(adversarialImages):
            # flag =1
            break
        fig.add_subplot(rows, columns, t)
        t = t + 1
        file = cv2.imread(adversarialImages[counterAdver])
        plt.imshow(file)
        counterAdver = counterAdver + 1
        plt.axis('off')
    
    if len(adversarialImages)-counterAdver<=columns:
        columns = len(adversarialImages)-counterAdver
    plt.savefig("Grids_1/comparison_"+str(f)+".jpg")
    f = f + 1
    fig = plt.figure(figsize=(x, y))
