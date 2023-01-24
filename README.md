# Лабораторная работа №4

При реализации программы использовались следующие библиотеки:
- torch
- torchvision
- Matplotlib
- Numpy
- Pandas
- datetime
- os
- glob
- xml.etree
- Pillow
- sklearn

Библиотеки glob, Pandas, xml.etree, Numpy, Pillow использовались для чтения и обработки датасета. Бибилиотеки torch и torchvision применялись для построения модели. В качестве архитектуры модели была выбрана YOLOv5m, ее загрузка производилась путем клонирования репозитория https://github.com/ultralytics/yolov5. Библиотека sklearn использовалась для деления датасета на тренировочную, валидационную и тестовую части.

# Алгоритм работы системы
Приложение работает в режиме реального времени с изображениями, на которых присутствуют люди. Изображения считываются при помощи библиотеки Pillow. Алгоритм работы программы следующий:
-	Изображение считывается, преобразуется к размеру 640 на 480 и конвертируется в файл с расширением .jpg;
-	Полученный файл отправляется на обработку в нейронную сеть YOLOv5m;
-	На выходе из нейронной сети получаем предсказанное значение и границы рамок для выделенных объектов, т.е. определенных моделью лиц людей;
-	Результат выводится в виде изначального изображения, а также изображения с выделенными нейронной сетью объектами. Выделение объектов происходит путем наложения рамок с обозначением вероятности отнесения объекта к определенной нейросетью классу.

# Обученная модель

Выборка, использованная при обучении модели, была разбита на три части: train, val и test. Размеры изображений были приведены к соотношению (640,480) и сохранены в формате .jpg.
За основу была выбрана модель YOLOv5m, которую мы самостоятельно обучали на тренировочной выборке датасета. Обучение модели заняло около трех часов. После обучения модели были получены следующие графики-результаты оценки модели:

![image](https://user-images.githubusercontent.com/79449892/214305589-8831c4b8-4aef-4362-adf3-ef32a5c2b4cc.png)

![image](https://user-images.githubusercontent.com/79449892/214305507-51d039da-fbfd-483b-b8cc-73b21f5364b7.png)

# Результаты работы модели
После запуска модели на тестовом датасете были получены следующие результаты:

Исходное изображение

![image](https://user-images.githubusercontent.com/79449892/214305736-adca0e57-654a-4f66-a02b-ca35137f6276.png)

Результат распознавания:

![image](https://user-images.githubusercontent.com/79449892/214305796-1a685bf2-a4f5-4d94-97d1-bf1f38eb1e9d.png)

Исходное изображение:

![image](https://user-images.githubusercontent.com/79449892/214305943-1f2d46c9-7bf4-4012-868d-82ed23b1c6a0.png)

Результат распознавания:

![image](https://user-images.githubusercontent.com/79449892/214305996-a182fec0-7738-4938-b76b-b3cc94806078.png)

Исходное изображение:

![image](https://user-images.githubusercontent.com/79449892/214307005-22f8aa8a-40f6-4f53-a1de-2209d9b06d3a.png)

Результат распознавания:

![image](https://user-images.githubusercontent.com/79449892/214307063-39e6ce9f-d9b5-48c0-8d05-e5ee2c0f30da.png)

Исходное изображение:

![image](https://user-images.githubusercontent.com/79449892/214307181-70e845b4-2503-4590-8388-04c5166b5240.png)

Результат распознавания:

![image](https://user-images.githubusercontent.com/79449892/214307273-76fbaa03-8259-457e-9d8f-89dff165ad62.png)

# Источники

https://github.com/ultralytics/yolov5

https://pytorch.org/docs/stable/index.html

https://pillow.readthedocs.io/en/stable/index.html

https://docs.python.org/3/library/xml.etree.elementtree.html
