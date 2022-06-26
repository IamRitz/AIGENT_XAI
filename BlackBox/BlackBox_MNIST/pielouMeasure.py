from cmath import log

def PielouMeaure(frequencies, num_classes):
    sum = 0
    for i in range(num_classes):
        sum=sum+frequencies[i]
    
    percents = []
    for i in range(num_classes):
        percents.append(float(frequencies[i])/sum)

    # print(percents)
    measure = 0
    for i in range(num_classes):
        if percents[i]==0:
            measure = measure+percents[i]*log(percents[i]+1)
        else:
            measure = measure+percents[i]*log(percents[i])
    return -1*measure/log(num_classes)

