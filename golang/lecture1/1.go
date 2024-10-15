package main

import (
	"fmt"
	"time"
)

const Hello = "Hello" // публичная константа, можно использовать вне пакета
const world = "world" // приватная константа, можно использовать только в этом пакете

func main() {
	var a = "initial" // можно не указывать тип
	fmt.Println(a)

	var b, c int = 1, 2 // можно опустить int

	// var d string   нельзя объявить переменную и не использовать

	var e bool // если мы не присваиваем ничего переменной, то тип указывается обязательно
	// по умолчанию ставится нулевое значение данного типа
	fmt.Println(b, c)
	fmt.Println(e)

	f := "apple" // ещё один вариант присваения
	fmt.Println(f)

	// a = 5  нельзя присвоить 5 строке, статическая типизация

	// var a = 5  нельзя снова инициализировать a

	// можно в ифе объявить переменную
	if num := 9; num < 0 {
		fmt.Println(num, "is negative")
	} else if num < 10 {
		fmt.Println(num, "has 1 digit")
	} else {
		fmt.Println(num, "has multiple digits")
	}

	// классический while
	i := 1
	for i <= 3 {

		i += 1
	}

	// классический for
	for i := 1; i <= 9; i++ {
		continue
	}

	// бесконечный цикл while true
	cnt := 0
	for {

		if cnt == 10 {
			break
		}
		cnt += 1
	}

	nums := []int{2, 3, 4}
	sum := 0

	// в качестве _ будет индекс, нельзя написать i и не использовать
	for _, num := range nums {
		sum += num
	}

	fmt.Println(sum)

	// switch case default

	switch time.Now().Weekday() {
	case time.Tuesday:
		fmt.Println("tuesday")
	case time.Saturday, time.Sunday:
		fmt.Println("weekend")
	default:
		fmt.Println("weekday")
	}

	t := time.Now()
	fmt.Println(t)
	switch { // в case сравнивается с true
	case t.Hour() < 23:
		fmt.Println("< 23")
		fallthrough // если не написать, то не провалится дальше. Т.е по умолчанию в каждом case стоит break
	case t.Hour() < 22:
		fmt.Println("< 22")
	default:
		fmt.Println("other")
	}

}
