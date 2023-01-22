# Теоретическая часть



# Описание разработанной системы


Функция нормализации при помощи функции normalize библиотеки OpenCV выглядит следующим образом:
```
def norm_cv(image, a, b):
    img = np.zeros((800, 800))
    norm_image = cv2.normalize(image, img, a, b, cv2.NORM_MINMAX)
    return norm_image
```
Функция нормализации написанная на Python вручную выглядит следующим образом:
```
def norm_python(image, a, b):
    norm_image = (a + (image - image.min())/(image.max() - image.min()) * (b - a)).astype(np.uint8)
    norm_image = np.round(norm_image)
    return norm_image
```
# Результаты работы системы


# Выводы по работе


# Источники

