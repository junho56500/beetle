import random
from bisect import bisect_left, bisect_right

class Search:

    def binary_recur(self, arr, query, s, e):
        if s > e:
            return None
        center = (s+e)//2
        if arr[center] == query:
            return center
        elif arr[center] < query:
            return self.binary_recur(arr, query, center+1, e)
        else:
            return self.binary_recur(arr, query, s, center-1)
#JUNO
    def binary(self, arr, query):
        s = 0
        e = len(arr)

        while s <= e:        
            center = (s+e)//2
            if arr[center] == query:
                break
            elif arr[center] < query:
                s = center + 1
            else:
                e = center - 1
        return center    
   
    def findArea(self, arr, r_query, l_query):
        r_ind = bisect_right(arr, r_query)
        l_ind = bisect_left(arr, l_query)
        return r_ind - l_ind
        
    
def main():
    arr = [i for i in random.sample(range(10), 10)]
    arr.sort()
    #arr.extend([5,3,4,7,1,0,0,0])  for count
    search = Search()
    s = 0
    e = len(arr)
    print(search.binary(arr, 7))

if __name__ == '__main__':
    main()

def binary(arr, num):
    s_idx = 0
    e_idx = len(arr)
    while s_idx >= e_idx:
        c_idx = int(s_idx + e_idx /2)
        if arr[c_idx] == num:
            return c_idx
        elif arr[s_idx] > num and arr[c_idx] > num:
            e_idx = c_idx
        elif arr[e_idx < num] and arr[c_idx] < num:
            s_idx = c_idx
    