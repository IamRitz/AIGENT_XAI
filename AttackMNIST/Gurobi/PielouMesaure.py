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

a = [16, 93, 0, 13, 2, 0, 1, 0, 148, 2]
b = [5, 12, 14, 24, 18, 11, 14, 6, 18, 13]
c = [7, 30, 22, 37, 19, 17, 27, 12, 34, 16]
d = [9, 26, 15, 42, 26, 20, 24, 6, 35, 23]
e = [1, 1, 1, 9, 0, 14, 46, 1, 123, 1]

print(PielouMeaure(a, len(a)))
print()
print(PielouMeaure(b, len(b)))
print()
print(PielouMeaure(c, len(c)))
print()
print(PielouMeaure(d, len(d)))
print()
print(PielouMeaure(e, len(e)))