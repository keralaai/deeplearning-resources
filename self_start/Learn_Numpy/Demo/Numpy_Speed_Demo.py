import numpy as np #numpy does not come bundled with python,make sure you have numpy installed
import time
#We will try to add two lists of same length, and see how much time that takes

array_length=1000

t1 = time.time()
a = range(array_length)
b = range(array_length)
c = [a[i] + b[i] for i in range(len(a)) ]
print("list addition took : %f" %(time.time() - t1))


#Now lets add two numpy arrays of same length, and see how much time that takes
t1 = time.time()
d = np.arange(array_length)
e = np.arange(array_length)
f= d + e

print("Numpy array addition took : %f"%(time.time() - t1))

print("This difference grows with increase in the length,try changing array_length to see it in action")





