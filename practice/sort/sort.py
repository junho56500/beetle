import random

class Sort:
                    
    def mixed(self,arr):
        for i in range(len(arr)):
            for j in range(i):
                if arr[i] < arr[j]:
                    arr[i],arr[j] = arr[j],arr[i]
        return(arr)
#JUNO    

                    
    def count(self, arr):
        count = [0]*10
        for i in arr:
            count[i] += 1
        
        arr_c = []
        for i, x in enumerate(count):
            for j in range(x):
                arr_c.append(i)

        return arr_c
            
    def merge(self, arr):
        if len(arr) < 2:
            return arr
        
        mid = len(arr) // 2
        left = self.merge(arr[:mid])
        right = self.merge(arr[mid:])

        arr_m = []
        l = r = 0
        while l < len(left) and r < len(right):
            if left[l] < right[r]:
                arr_m.append(left[l])
                l += 1
            else:
                arr_m.append(right[r])
                r += 1

        arr_m += left[l:]
        arr_m += right[r:]
        return arr_m                
                    
        
    
def main():
    arr = [i for i in random.sample(range(10), 10)]
    sort = Sort()
    #arr.extend([5,3,4,7,1,0,0,0])  for count
    print(sort.insertion_sort(arr))

if __name__ == '__main__':
    main()
    