x =  5 * 10**7
for i in range(x):
    try:
        b = (i * 2) / 0
    except Exception as ex:
        pass
    
# Сравнить время цикла в этом файле, где бросается исключение и в файле 1.py, без исключения.