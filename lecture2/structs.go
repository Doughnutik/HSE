package main

type book struct {
	title  string `json:"title"`
	Author string `json:"author, omitempty"`
	pages  int    `json:"pages"`
	// зелёное - теги
	// по умолчанию заполняются нулевыми значениями
}

func main() {
	var book_one book         // первый вариант инициализации
	book_one.Author = "James" // публичное поле, видно за пределами пакета main
	book_one.title = "name"   // приватное поле, не видно за пределами пакета main

	var book_two book = book{"later", "Feddi", 50} // второй вариант инициализации

	var book_three book = book{pages: 7, title: "title"}

	println(book_one.pages, book_one.title, book_one.Author)

	println(book_two.pages, book_two.title, book_two.Author)

	println(book_three.pages, book_three.title, book_three.Author)
}
