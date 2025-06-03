



def ex1(n,k):
    cnt = 0
    while True:
        if n == 1:
            break
        if n%k != 0:
            n -= 1
        else:
            n /= k
        cnt += 1
        
    return cnt

def main():
    cnt = ex1(17,4)
    print(cnt)

if __name__ == '__main__':
    main()