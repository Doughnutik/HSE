#include <assert.h>

void swap(int& a, int& b){ 
    int t = a;
    a = b;
    b = t;
}

int median(int a, int b, int c) {
    if (a > b) swap(a, b);
    if (a > c) swap(a, c);
    if (b > c) swap(b, c);
    return b;
}

int main(){ 
    assert(median(3, 2, 1) == 2);
    assert(median(5, -1, 2) == 2);
    assert(median(3, 3, 5) == 3);
}
