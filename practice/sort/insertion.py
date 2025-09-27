# A helper function to print the contents of a list.
# // Time Complexity: O(n^2)
def print_list(arr):
    """
    Prints the elements of a list.

    Args:
        arr: The list to be printed.
    """
    for element in arr:
        print(element, end=" ")
    print()

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    
# Example usage:
if __name__ == "__main__":
    data = [12, 11, 13, 5, 6]
    
    print("Original list:")
    print_list(data)
    
    insertion_sort(data)
    
    print("Sorted list (Insertion Sort):")
    print_list(data)      