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
e = [3, 12, 2, 8, 1, 24, 58, 1, 146, 17]
f = [5, 10, 9, 8, 2, 32, 53, 7, 121, 28]
g = [5, 10, 8, 8, 2, 37, 52, 7, 118, 28]
h = [5, 5, 11, 10, 10, 27, 49, 19, 122, 17]
print(PielouMeaure(a, len(a)))
print()
print(PielouMeaure(b, len(b)))
print()
print(PielouMeaure(c, len(c)))
print()
print(PielouMeaure(d, len(d)))
print()
print(PielouMeaure(e, len(e)))
print()
print(PielouMeaure(f, len(f)))
print()
print(PielouMeaure(g, len(g)))
print()
print(PielouMeaure(h, len(h)))