from time import time
# from attackMethod import attack
from newatm import attack

t1 = time()
count  = attack()
t2 = time()
print("TIME TAKEN IN GENERATION OF ABOVE EXAMPLES: ", (t2-t1)/count, "seconds.")
