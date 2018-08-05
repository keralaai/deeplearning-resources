
#heres how we can create a matrix(a 2d array) using lists,simply create a list of lists
matrix = []
inner = []
for j in range(5):
    for i in range(10):
        inner.append(i+j*10)
    matrix.append(inner)
    inner = []
print("List :")    
print(matrix)


#now lets try doing the same thing with Numpy
import numpy as np #numpy does not come bundled with python,make sure you have numpy installed
np_array = np.arange(50).reshape(5,10)


print("\n Numpy array : \n {} ".format(np_array))


print('''\n As u can see we just needed two lines to create a Numpy array while it took  nested loops in the case of lists
also ''')