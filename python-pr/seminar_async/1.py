import threading
from time import sleep
from random import randint, seed

def func(x: int, lock: threading.Lock):
    sum = 0
    cnt = 10000000
    for i in range(cnt):
        sum += i * x
        x += i
    with lock:
        print(x)
        
def main():
    l = 1
    r = 100
    cnt = 20
    seed(43)
    values = [randint(l, r) for _ in range(cnt)]
    # print(values)
    lock = threading.Lock()
    threads = []
    for value in values:
        threads.append(threading.Thread(target=func, args=[value, lock]))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
if __name__ == "__main__":
    main()