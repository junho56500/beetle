import random

class Sort:
    def selection(self,arr):
        min_ind=0
        min_j=0
        for i in range(len(arr)):
            min_j = arr[i]
            for j in range(i+1, len(arr)):
                if arr[j] < min_j:
                    min_j = arr[j]
                    min_ind = j
            arr[i],arr[min_ind] = arr[min_ind],arr[i]
        return arr

#JUNO    
    def selection_modern(self,arr):
        for i in range(len(arr)):
            min_value = min(arr[i:])
            min_ind = arr.index(min_value)
            arr[i],arr[min_ind] = arr[min_ind],arr[i]
        return arr
#JUNO    
    def bubble(self,arr):
        for i in range(len(arr)):
            for j in range(len(arr)-1-i):
                if arr[j] > arr[j+1]:
                    arr[j],arr[j+1] = arr[j+1],arr[j]
        return arr
                    
    def mixed(self,arr):
        for i in range(len(arr)):
            for j in range(i):
                if arr[i] < arr[j]:
                    arr[i],arr[j] = arr[j],arr[i]
        return(arr)
#JUNO    
    def insertion_sort(self, arr):
        """
        Sorts a list in ascending order using the Insertion Sort algorithm.

        The algorithm iterates through the list, taking one element at a time
        from the unsorted part and inserting it into its correct position
        in the sorted part.

        Time Complexity: O(n^2)
        Space Complexity: O(1)

        Args:
            arr: The list of integers to be sorted.
        """
        # The first element is considered sorted, so we start from the second element (index 1).
        for i in range(1, len(arr)):
            # Store the current element to be inserted in the sorted part.
            key = arr[i]
            
            # Start comparing the key with elements in the sorted part, from right to left.
            j = i - 1

            # Shift elements of the sorted part that are greater than the key,
            # one position to the right to make space for the key.
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            
            # Place the key at its correct position.
            arr[j + 1] = key
    
    #more follow principal
    def insertion_sort(self, arr):
        """
        Sorts a list using the Insertion Sort algorithm.
        """
        # Traverse through 1 to n-1
        # assuming the first element (index 0) is already sorted
        for i in range(1, len(arr)):
            key = arr[i]  # Current element to be inserted
            j = i - 1     # Index of the last element in the sorted part

            # Move elements of arr[0..i-1], that are greater than key,
            # to one position ahead of their current position
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key # Place key at its correct position
        return arr
    
#JUNO    
    def quick_modern(self,arr):
        if len(arr) <= 1:
            return arr
        
        piv = arr[0]
        tail = arr[1:]
        
        left_side = [x for x in tail if x <= piv]
        right_side = [x for x in tail if x > piv]
        
        return self.quick(left_side) + [piv] + self.quick(right_side)
                    
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
    