#include <iostream>
using namespace std;

int main() {
    int y = 2;
    std::cout << y + "aboba\n";  // можем прибавить число к строке и получить сдвиг начала
    std::cout << y + 'a';  // а вот прибавив к чару, получили другой чар на 'y' больше
}