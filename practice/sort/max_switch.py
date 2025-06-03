import os
import random

def ex1(array1, array2, K):
    
    # N = len(array1)
    
    # if N != len(array2):
    #     return None

    array1.sort()
    array2.sort(reverse = True)
    
    for i in range(K):
        array1[i], array2[i] = array2[i], array1[i]
    
    return array1

def main():
    arr1 = [i for i in random.sample(range(10), 10)]
    arr2 = [i for i in random.sample(range(10), 10)]
    
    print(ex1(arr1, arr2, 3))

if __name__ == '__main__':
    main()