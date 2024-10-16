package main

import "fmt"

func main() {
	// инициализация слайсов
	var s1 []int
	s2 := []int{1, 2, 3}
	s3 := make([]int, 2, 3) // make([]int, len, cap)
	// len, cap - длина слайса и capacity, len <= cap

	s1 = append(s1, 5, 6, 7) // Возвращается копия с новыми данными. в GO принято сохранять новые данные в ту же самую переменную
	s1 = append(s1, s2...)
	s1 = append(s1, s3...)
	for _, num := range s1 {
		print(num)
		print(" ")
	}
	println()
	println(len(s1))

	s4 := []int{1, 2, 3, 4, 5}
	s5 := s4[1:4]
	s5[0] = 10 // заметим, что также поменяются данные в s4
	s4[2] = 11 // s5 тоже поменяется
	fmt.Println(s4)
	fmt.Println(s5)
	// Т.е. слайс ссылается на одну и ту же память, новая память не выделяется

	words := []string{"a", "b", "c", "d", "e"}
	fmt.Println(len(words), cap(words))      // длина и capacity
	words = append(words[1:2], words[2:]...) // для удаления элементов используем реслайс
	fmt.Println(words)
	words = append(words[1:2], words[3:4]...)
	fmt.Println(words)
	fmt.Println(len(words), cap(words))
}
