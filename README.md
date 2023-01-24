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

Библиотеки glob, Pandas, xml.etree, Numpy, Pillow использовались для чтения и обработки датасета. Бибилиотеки torch и torchvision применялись для построения модели. В качестве архитектуры модели была выбрана YOLOv5s, ее загрузка производилась путем клонирования репозитория https://github.com/ultralytics/yolov5.

# Алгоритм работы системы
Приложение работает в режиме реального времени с изображениями, на которых присутствуют люди. Изображения считываются при помощи библиотеки OpenCV (этого у нас нет, но можно добавить быстро). Алгоритм работы программы следующий:
-	Изображение считывается, преобразуется к размеру 640 на 480 и конвертируется в файл с расширением .jpg;
-	Полученный файл отправляется на обработку в нейронную сеть YOLOv5s;
-	На выходе из нейронной сети получаем предсказанное значение и границы рамок для выделенных объектов, т.е. определенных моделью лиц людей;
-	Результат выводится в виде изначального изображения, а также изображения с выделенными нейронной сетью объектами. Выделение объектов происходит путем наложения рамок с обозначением вероятности отнесения объекта к определенной нейросетью классу.

# Обученная модель

За основу была выбрана модель YOLOv5s, которую мы самостоятельно обучали на тренировочной выборке нашего датасета. Обучение модели заняло около трех часов. После обучения модели были получены следующие графики-результаты оценки модели:

![image](https://user-images.githubusercontent.com/79449892/214260697-8455df95-f2ab-44ef-8e17-60434663b3eb.png)

![image](https://user-images.githubusercontent.com/79449892/214260747-6067e4d4-33b0-4840-a239-d3d7df4e04f2.png)

# Результаты работы модели
После запуска модели на тестовом датасете были получены следующие результаты:

Исходное изображение

![image](https://user-images.githubusercontent.com/79449892/214261350-3a68436d-f1fd-41c0-9c1a-f729a9cfd978.png)

Результат распознавания:

![image](https://user-images.githubusercontent.com/79449892/214261402-ea1d8ba7-dd43-4586-9fec-7ca2e516cfae.png)

Исходное изображение:

![image](https://user-images.githubusercontent.com/79449892/214261605-b57445af-b892-416c-9fc1-58ab4ec2a36a.png)

Результат распознавания:

![image](https://user-images.githubusercontent.com/79449892/214261659-b1cbbfde-c422-4474-abfb-4a54866f319c.png)

Исходное изображение:

![image](https://user-images.githubusercontent.com/79449892/214261737-d2c16496-258f-48aa-9fcd-3e67b9e4ebc2.png)

Результат распознавания:

![image](https://user-images.githubusercontent.com/79449892/214261827-54d7eb6b-315b-4e45-a31b-dfb3ec03687c.png)

Исходное изображение:

![image](https://user-images.githubusercontent.com/79449892/214261954-85d3b97c-1a1a-4e3f-8241-8f3e54efc0e0.png)

Результат распознавания:

![image](https://user-images.githubusercontent.com/79449892/214262042-f5c5f630-9278-4b78-aeab-6d380a648d80.png)

# Источники
https://github.com/ultralytics/yolov5
https://pytorch.org/docs/stable/index.html
https://pillow.readthedocs.io/en/stable/index.html
https://docs.python.org/3/library/xml.etree.elementtree.html
