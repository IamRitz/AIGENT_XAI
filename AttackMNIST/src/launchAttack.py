from time import time
from attackMethod import generate

t1 = time()
count = generate()
t2 = time()
print("TIME TAKEN IN GENERATION OF ABOVE EXAMPLES: ", (t2-t1)/count, "seconds.")
