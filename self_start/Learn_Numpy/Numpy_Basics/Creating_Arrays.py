import numpy as np

# 1-d array

a=np.array([2,4,6,8]) #creates an array with given list
print(a)
a=np.arange(2,10,2)   #similar to range() 2 to 9 in steps of 2 
print(a)
a=np.linspace(2,8,4) #creates an array with 4 elements equally spaced between 2 and 8
print(a)



#now lets try 2-d arrays
print("\n\n 2-d array\n using list of tuples")

a=np.array([(1,2,3),(4,5,6),(7,8,9)]) #creates a 2d array
print(a)

print('using list of lists')
a=np.array([[1,2,3],[4,5,6],[7,8,9]]) #creates a 2d array
print(a)

a=np.array([(1,2,3),(4,5,6),(7,8,9)],dtype=np.int8) #specify the datatype



print("zeros")
a=np.zeros((3,4))       #creates an array of zeroes with 3 rows and 4 columns
print(a)
a=np.zeros(10)       #creates an array of 10 zeros
print(a)

print("\n ones")
a=np.ones((3,4))       #creates an array of zeroes with 3 rows and 4 columns
print(a)
a=np.ones(10)       #creates an array of 10 ones
print(a)


print("\n Random values")
a=np.random.random((3,2))      #array of random values
print(a)
a=np.random.randint(0,20,6)      #array of random values
print(a)