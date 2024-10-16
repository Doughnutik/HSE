package main

import "fmt"

func voidPlus(a int, b int) {
	// return a + b  если не указан тип, то он void
	return
}

func plus(a int, b int) int {
	// return  нельзя вернуть "ничего", тогда не надо указывать возвращаемый тип
	return a + b
}

func plusPlus(a, b, c int, d, e, f string) int {
	// можно подряд писать аргументы одного типа
	s := d + e + f // можно складывать строки
	fmt.Println(s)
	return a + b + c
}

func multiple() (int, string, bool) {
	return 5, "aboba", false // можно возвращать несколько значений
}

func main() {
	// fmt.Println(voidPlus()) не скомпилится, так как ничего не возвращается
	fmt.Println("5 + 6 =", plus(5, 6))
	fmt.Println(plusPlus(1, 2, 3, "a", "b", "c")) // важно, что строки только в двойных кавычках

	a, b, _ := multiple()

	fmt.Println(a, b) // вывод через пробел
	fmt.Print(a)      // вывод слитно
	fmt.Print(b)

	_, _c_, _ := multiple() // можно пропускать любые аргументы с помощью _, но обязательно именно _ (_c_ уже переменная)

	fmt.Println(_c_)
}
