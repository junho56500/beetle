import collections import deque


def stack():
    a = []
    a.append(1)
    a.append(2)
    a.append(3)
    a.pop()
    a.append(4)
    print(a)

def queue():
    a = deque()
    a.append(1)
    a.append(2)
    a.append(3)
    a.popleft()
    a.append(4)
    print(a)
    

def main():
    stack()
    queue()

if __name__ == '__main__':
    main()