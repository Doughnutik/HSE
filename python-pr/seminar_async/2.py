import asyncio
from random import randint, seed

async def func(x: int):
    sum = 0
    cnt = 10000000
    for i in range(cnt):
        sum += i * x
        x += i
    print(x)
        
async def main():
    l = 1
    r = 100
    cnt = 20
    seed(43)
    values = [randint(l, r) for _ in range(cnt)]
    # print(values)
    tasks = []
    for i in values:
        tasks.append(asyncio.create_task(func(i)))
    await asyncio.gather(*tasks)
    
    
if __name__ == "__main__":
    asyncio.run(main())