from time import time
from attackMethod import attack

t1 = time()
attack()
t2 = time()
print("TIME TAKEN IN GENERATION OF ABOVE EXAMPLES: ", (t2-t1)/count, "seconds.")
