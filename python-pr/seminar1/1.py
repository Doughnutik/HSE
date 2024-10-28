class A:
    public_number = 0
    _protected_number = 0
    __private_number = 0

    def __init__(self):
        self.number = 5
        self._protected_number = 10
        self.__private_number = 15
    def __str__(self):
        return str(self.number) + "\n" + str(self._protected_number)
    def __repr__(self):
        return str(self.number) + "\n" + str(self.__private_number)
    def __del__(self):
        # типо деструктор, но он автоматический в питоне и в целом не нужен
        return
    

a = A()
print(a._protected_number) # печально, что можно вызвать protected поле

class SuperAnimal:
    hands = 0
    def __init__(self, x):
        self.hands = x
        
    def voice(self):
        print("hello")

class Animal():
    _legs = 4
    def __init__(self, x):
        self._legs = x
    
    def voice(self):
        print("abstract")
    
class Cat(Animal, SuperAnimal):
    def __init__(self, x):
        super().__init__(x)
        
    def voice(self):
        print("мяу")
        
c = Cat(5)
print(c._legs, c.hands)
a = Animal(7)
a.voice()
c.voice()


# Абстрактные методы и классы
import abc
class B(abc):
    
    @abc.abstractmethod
    def voice():
        pass
    
b = B()