import chardet

file_path = 'data\\test100.csv'
with open(file_path, 'rb') as file:
    result = chardet.detect(file.read())
    print(result)
