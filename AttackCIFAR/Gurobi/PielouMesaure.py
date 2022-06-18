from cmath import log

def PielouMeaure(frequencies, num_classes):
    sum = 0
    for i in range(num_classes):
        sum=sum+frequencies[i]
    
    percents = []
    for i in range(num_classes):
        percents.append(float(frequencies[i])/sum)

    measure = 0
    for i in range(num_classes):
        if percents[i]==0:
            measure = measure+percents[i]*log(percents[i]+1)
        else:
            measure = measure+percents[i]*log(percents[i])
    return -1*measure/log(num_classes)

beforeChangingUpdateStatement = [5, 69, 32, 63, 30, 5, 39, 29, 6, 22]
afterChangingUpdateStatement = []

print()
print(PielouMeaure(beforeChangingUpdateStatement, len(beforeChangingUpdateStatement)))

print()
print(PielouMeaure(afterChangingUpdateStatement, len(afterChangingUpdateStatement)))