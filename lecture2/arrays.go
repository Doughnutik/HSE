package main

func main() {

	var a = [5]int{1, 2, 3, 4, 5}
	var b [6]int // [5]int, [6]int - полноценные разные типы

	c := [...]int{1, 2, 3} // автоматически вычислит размер

	// d := [...]int   так нельзя писать

	for _, num := range a {
		print(num)
		print()
	}
	println()
	for _, num := range b {
		print(num)
		print()
	}
	println()
	for _, num := range c {
		print(num)
		print()
	}
	println()

	// x := 3
	println(c[0]) // нумерация с нуля
	// println(c[3]) на этапе компиляции ловит выход за границы массива
	// Однако println(c[x]) вызовет RE
	// println(c[-1])  отрицательных индексов нет
}
