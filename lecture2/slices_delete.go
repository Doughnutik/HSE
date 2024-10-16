package main

func main() {
	// Если нам нужно удалить элемент и нам не важен порядок, эффективней это делать свапом с последним

	const length, cnt int = 1e7, 9e6
	a := make([]int, length, length)
	ind := 15
	for i := 0; i <= cnt; i++ {
		a[ind] = a[len(a)-1]
		a = a[:len(a)-1]

		// a := append(a[:ind], a[ind+1:]...)

		ind = (ind * 19) % len(a)
	}
	// Можно сравнить время работы в зависимости от варианта удаления, разница колоссальная
}
