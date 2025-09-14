#JUNO    
#// Time Complexity: Average O(n log n), Worst-case O(n^2)

def quick_modern(self,arr):
    if len(arr) <= 1:
        return arr
    
    piv = arr[0]
    tail = arr[1:]
    
    left_side = [x for x in tail if x <= piv]
    right_side = [x for x in tail if x > piv]
    
    return self.quick(left_side) + [piv] + self.quick(right_side)