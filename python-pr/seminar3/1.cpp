#include <iostream>

struct worker 
{
    worker& process(int& i)
    {
        i = 185;
        return *this;
    }
    worker& print_result(const int& i)
    {
        std::cout <<"result: "<< i << std::endl;
        return *this; 
    }
};

int main()
{
    int data = 0;
    worker w;
    w.process(data).print_result(data+2);
}