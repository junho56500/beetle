import random
from bisect import bisect_left, bisect_right

class Search:

    def func(arr, m):
        n = len(arr)
        s = 0
        e = max(arr)
        tot_m = 0
        while s >= e:
            center = (e-s)//2
            for i in arr:
                j = i-center
                if j>0:
                    tot_m += j
            if tot_m == m:
                break
            elif tot_m < m:
                s = center + 1
            else:
                e=center-1
                
                
        
    
def main():
    arr = [i for i in random.sample(range(10), 10)]
    arr.extend([5,3,4,7,1,0,0,0])
    search = Search()
    print(search.func(arr, 7))

if __name__ == '__main__':
    main()