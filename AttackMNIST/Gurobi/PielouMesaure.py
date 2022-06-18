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

with192 = [6, 11, 8, 7, 2, 23, 56, 6, 123, 31]

print(PielouMeaure(with192, len(with192)))
print()

with384 = [6, 11, 8, 7, 2, 23, 56, 6, 123, 31]

print(PielouMeaure(with384, len(with384)))
print()