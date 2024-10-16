package main

import "fmt"

func sum(nums ...int) int { // переменное кол-во аргументов. ... аргумент может быть только последним в списке
	total := 0

	for _, num := range nums {
		total += num
	}

	return total
}

func main() {
	nums := []int{1, 2, 3, 4, 5}

	fmt.Println("1 + 2 + 3 + 4 + 5 =", sum(nums...))
	fmt.Println("1 + 2 + 3 =", sum(1, 2, 3))
}
