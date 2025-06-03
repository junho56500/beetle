


def count3(num):
    count = 0
    for i in range(num):
        if i%10 == 3 or int(i/10) == 3:
            count += 1
    return count
        
        

def ex1(h):
    if h > 3:
        cnt_h  = 0
    else:
        cnt_h = 3600    
    cnt_m = count3(60)
    print(cnt_m)
    cnt_s = count3(60)
    
    return int( cnt_h + cnt_m * cnt_s)

def main():
    ret = ex1(2)
    print(ret)

if __name__ == '__main__':
    main()