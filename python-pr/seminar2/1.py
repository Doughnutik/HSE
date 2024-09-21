def func(x) -> int:
    return "aboba"

# print(func(5))



a = [1, 2, 3]



def sum(x):
    def add(y):
        def multy(z):
            return x + (y * z)
        return multy
    return add
    

print(sum(5)(6)(7))

def cur_func(x):
    return x**2

a = [1, 2, 3, 4, 5]
b = list(map(lambda x: x**2, a))
c = list(map(cur_func, a))
print(b)
print(c)

a = ['mama', 'papa', 'sister']
b = list(filter(lambda x: len(x) == 4, a))
print(b)


doubles = [lambda x: x * 2 for x in range(5)]
print(doubles)
print(doubles[2](5))
print(doubles[3](5))
print("\n")

doubles = [lambda y: x * y for x in range(5)]
print(doubles[2](5))
print(doubles[3](5))