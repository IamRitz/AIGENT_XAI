from time import time
from attackMethod import generate

t1 = time()
count = generate()
t2 = time()
# print("Total No Exp: ", count[0])
# print("Total No Sig: ", count[1])
# print("Total No Pair: ", count[2])
# print("Total No UB: ", count[3])
print("TIME TAKEN IN GENERATION OF ABOVE EXAMPLES: ", (t2-t1)/count, "seconds.")
