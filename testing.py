import chardet

file_path = 'data\\processed_training_turkish_2.csv'
with open(file_path, 'rb') as file:
    result = chardet.detect(file.read())
    print(result)
