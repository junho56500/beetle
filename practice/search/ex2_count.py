import random
from bisect import bisect_left, bisect_right

class Search:

    def func(self, arr, x):
        right_id = bisect_right(arr,x)
        left_id = bisect_left(arr,x)
        print(right_id, left_id)
        return right_id - left_id
                
        
    
def main():
    arr = [i for i in random.sample(range(10), 10)]
    arr.extend([5,3,4,7,1,0,0,0])
    arr.sort()
    print(arr)
    search = Search()
    print(search.func(arr, 7))

if __name__ == '__main__':
    main()