# Learn Numpy

## What is Numpy?
Numpy is a library for scientific computing in python. It provides a multidimensional array object and tools for manipulating and processing these arrays.NumPy can also be used as an efficient multi-dimensional container of generic data.

## Why Numpy?
Why use Numpy when we already have lists.
 * 3 reasons
    * **Less Memory** - A Numpy array uses much less space compared to a list as demonstrated in [Numpy_Memory_Demo](Demo/Numpy_Memory_Demo.py) . 
    * **Speed** - A Numpy array uses much faster compared to a list as demonstrated in [Numpy_Speed_Demo](Demo/Numpy_Speed_Demo.py) . 
    * **Convenience**- It is much more convinient to perform operations in Numpy than in lists as demonstrated in [Numpy_Convenience_Demo](Demo/Numpy_Convenience_Demo.py) . 

## Installing Numpy

[Installation options](https://www.scipy.org/install.html)

[PyCharm users](https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html)


# Getting started

## Numpy Basics

Import numpy to your project

    import numpy as np

creating a numpy array

    a=np.array([2,4,6,8]) #creates an array with given list
    a=np.arange(1,10,2)   #similar to range()
    a=np.linspace(2,10,4) #creates an array with 4 elements between 2 and 10

see [creating arrays](Numpy_Basics/Creating_Arrays.py) for more examples including 2d arrays

#### Numpy datatypes
All elements in a numpy array have the same datatype

Data type |	Description
-------|-------
bool_ |	Boolean (True or False) stored as a byte
int_ |Identical to C int (normally int32 or int64)
intp |	Integer used for indexing (same as C ssize_t; normally either int32 or int64)
int8 |	Byte (-128 to 127)
int16 |	Integer (-32768 to 32767)
int32 |	Integer (-2147483648 to 2147483647)
int64 |	Integer (-9223372036854775808 to 9223372036854775807)
uint8 |	Unsigned integer (0 to 255)
uint16 |	Unsigned integer (0 to 65535)
uint32 |	Unsigned integer (0 to 4294967295)
uint64 |	Unsigned integer (0 to 18446744073709551615)
float_ |	Shorthand for float64.
float16 |	Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
float32 |	Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
float64 |	Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
complex_ |	Shorthand for complex128.
complex64 |	Complex number, represented by two 32-bit floats (real and imaginary components)
complex128| 	Complex number, represented by two 64-bit floats (real and imaginary components)

Source: [Docs](https://docs.scipy.org/doc/numpy-1.10.1/user/basics.types.html#array-types-and-conversions-between-types)

#### Getting Values

    a[0][1]  #both return the same element (The second element of the first row)
    a[0,1]   

see more [examples](Numpy_Basics/Reading_Arrays) on getting specific set of elements based on conditions in [Reading Arrays](Numpy_Basics/Reading_Arrays)


#### Basic Operations

    #these are properties and so no paranthesis
    a.shape #returns shape of array for example (3,2) meaning  3 rows and 2 columns
    a.size  #No. of elements in the array
    a.dtype #returns the datatype of the numpy array,all elements have the same datatype

    a.sum() #returns sum of all elements
    a.sum(axis=0) #sum all columns
    a.sum(axis=1)  #sum all rows

More operations : [Basic Operations](Numpy_Basics/Basic_operations)