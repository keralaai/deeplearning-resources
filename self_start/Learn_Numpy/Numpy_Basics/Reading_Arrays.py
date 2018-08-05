import numpy as np

a=np.array([(1,2,3),(4,5,6),(7,8,9)]) #creates a 2d array
print(a)

print("\n getting one specific element")
print(a[0][1])
print(a[0,1])

print("\n getting set of elements")
print(a[0:,2]) #prints the third(index 2) element in each row
print(a[1,1:]) #prints all elements in second row(index 1) except index 0
print(a[:,1:3])#prints second and third elements(index 1,2) of all rows



