



def ex1(input):
    tot = 0
    input = list(map(int,input))
    if len(input) < 2:
        return input[0]
    if input[0] > 1 and input[1] > 1:
        tot = input[0] * input[1]
    else:
        tot = input[0] + input[1]
        
    for i in range(2, len(input)):
        a = int(input[i])
        if a > 1:
            tot *= a
        else:
            tot += a
    return tot

def main():
    ret = ex1('17460')
    print(ret)

if __name__ == '__main__':
    main()