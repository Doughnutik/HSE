x = 3

match x:
    case "test":
        print("hello")
    case 3:
        print("world")

try:
    a = 5
    b = "aboba"
    print(a + b)
except:
    print("питон определяет типы во время runtime")


y = 5
x = str(y)
print(x, type(x))

x: int = "aboba"
x: float = 15.5

def func(x: int, y: float) -> float:
    return x + y

print(func(1, 5.5))


def func() -> None:
    # функция ничего не возвращает
    pass

print(func())


from typing import Union, Optional, Callable, List

x: Optional[int]
y: Union[int, float]

x = 123
x = None
x = "aboba" # не хорошо

# Но уже Optional не обязателен, можно писать через |
x: int | float
x = 123
x = None
print(x, type(x))


def call_func(a: float, b: int) -> int:
    return int(a + b)


c: Callable[[float, int], int] = call_func(1.1, 5)  #TODO !!!!!



mylist: List[int | str] = [1, "aboba", None]


from typing import Generic, TypeVar

T = TypeVar('T')

def reverse(x: List[T]) -> List[T]:
    return x[::-1]

