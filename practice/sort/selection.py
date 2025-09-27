
#// Time Complexity: O(n^2)

# def selection(arr):
#     min_ind=0
#     min_j=0
#     for i in range(len(arr)):
#         for j in range(i+1, len(arr)):
#             if arr[j] < arr[min_ind]:
#                 min_ind = j
#         arr[i],arr[min_ind] = arr[min_ind],arr[i]
#     return arr


def selection(arr):
    min_i = 0
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_i]:
                min_i = j
        arr[i],arr[min_i] = arr[min_i],arr[i]
    return arr

def main():
    a = [8,2,4,5,3,8,9,1,0]
    ret = selection(a)
    print(ret)
    
    
if __name__ == "__main__":
    main()
                