



def ex1(input, thr_scar, thr_num):
    a = list(map(int, str(input)))
    cnt_s = 0
    for i in a:
        if i > thr_scar:
            cnt_s += 1
    
    return int(cnt_s / thr_num)

def main():
    ret = ex1(17460655,3,3)
    print(ret)

if __name__ == '__main__':
    main()