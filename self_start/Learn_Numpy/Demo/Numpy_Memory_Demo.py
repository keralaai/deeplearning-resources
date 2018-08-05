import numpy as np #numpy does not come bundled with python,make sure you have numpy installed
import sys

numpyarray=np.arange(100)
print("This is a Numpy Array")
print(numpyarray)


my_list=range(100)
print("\n\nThis is a list with same content :")
print(list(my_list))

print("\n\nThe {} - dimensional numpy array of type {} takes up around {} bytes, whereas the list took around {} bytes"
        .format( numpyarray.ndim, numpyarray.dtype,numpyarray.size*numpyarray.itemsize ,sys.getsizeof(1)*len(my_list)))

print('''\nIn fact u can further reduce the data consumption if u specify the datatype\n
since we dont expect numbers larger than 100 here we can save some memory,if we use int8 instead of int64\n
  ''')  

print("\n\n So this new numpy array holds the same data :")
numpyarray=np.arange(100).astype(np.int8)
print(numpyarray)
print("\nBut Now the Numpy array takes only around {} bytes\n\n\n ".format(numpyarray.size*numpyarray.itemsize))
