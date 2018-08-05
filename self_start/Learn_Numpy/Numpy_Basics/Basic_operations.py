import numpy as np

a=np.array([(1,2,3),(4,5,6),(7,8,9)])
print("The array :\n{}\n\n".format(a))

print(a.sum()) #returns sum of all elements
print(a.sum(axis=0)) #sum all columns
print(a.sum(axis=1))  #sum all rows


print("\n Reshape")

a=np.ones((1,10))
print("Array")
print(a)

print("reshaped array")
print(a.reshape(2,5))


print("\n Operations")
a=np.array([(1,2,3),(4,5,6),(7,8,9)])
b=np.array([(1,2,3),(4,5,6),(7,8,9)])

print("Logical")
print(a<5)
print("Arithematic")
print(" Addition: \n {} \n\n Multiplication : \n{} \n\n Division :\n{} \n\n Subtraction:\n{}"
.format(a+b,a*b,a/b,a-b))


print("\n\n\n Variance:")
print(a.var())
print("\n\n\n Standard deviation:")
print(a.std())


print("\n\n\n Randomshuffle:")
np.random.shuffle(a)
print(a)

a=np.arange(100)
# a must be 1d for this
print("\n\n\n get a random value:")
print(np.random.choice(a))

