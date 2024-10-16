package main

func pow(x *int) *int {
	*x = *x * *x
	return x
}

func main() {
	var a int = 5
	var b *int = &a

	println(b, *b)

	ptr1 := new(int) // not nil

	// var ptr2 *int // nil
	// println(*ptr2) // panic, так как nill

	// panic: runtime error: invalid memory address or nil pointer dereference
	// [signal SIGSEGV: segmentation violation code=0x2 addr=0x0 pc=0x1009026c0]

	// goroutine 1 [running]:
	// main.main()
	//         /Users/opyat-ne-zavoditsya/hse/code/pr/golang/lecture2/1.go:17 +0x60
	// exit status 2

	a = 6
	ptr3 := &a

	println(*ptr1) // okay, будет 0
	println(*ptr3)

	println(*pow(&a))
	println(a)
}
