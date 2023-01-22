# Теоретическая часть
Классификация изображений - это процесс извлечения информации о классах объектов из многоканального растрового изображения.
В данной работе была произведена классификация изображений животных при помощи трех типов нейронных сетей: AlexNet, ResNet50 и InceptionV3.

В общем виде архитектура нейронной сети AlexNet выглядит следующим образом:

![image](https://user-images.githubusercontent.com/79449892/213929629-89aca09a-15ee-41ea-8c3b-ea647e20ea29.png)

В общем виде архитектура нейронной сети ResNet50 выглядит следующим образом:

![image](https://user-images.githubusercontent.com/79449892/213929654-898738f8-5a03-449c-a68c-bf9a1ea785f4.png)

В общем виде архитектура нейронной сети InceptionV3 выглядит следующим образом:

![image](https://user-images.githubusercontent.com/79449892/213929496-7686204c-dc51-4a30-9d1f-3bb51639152f.png)

# Описание разработанной системы
Система, построенная на основе предобученных сетей трех видов со стандартными весами, использовалась для классифифкации изображений животных. Для проверки точности модели был собран небольшой датасет (51 изображение), для которого вручную в соответствии с классами выборки, на которой проводилось обучение модели, были размечены истинные классы. По этим данным было проведено сравнение истинных и полученных классов и рассчитаны top1 и top5 точности.
# Пример работы системы
Изображение:

![6bb8056eaa](https://user-images.githubusercontent.com/79449892/213930092-cae3a693-a7be-4509-80b7-0c989d55c5a3.jpg)

Результат классификации при помощи модели AlexNet:

![image](https://user-images.githubusercontent.com/79449892/213930188-4aeb4aac-7e1d-4689-b6f9-a884972757d8.png)

Результат классификации при помощи модели Resnet50:

![image](https://user-images.githubusercontent.com/79449892/213930388-cbb66e4b-d4d6-4fc1-9112-23508fbc1783.png)

Результат классификации при помощи модели InceptionV3:

![image](https://user-images.githubusercontent.com/79449892/213930424-d152b22a-740d-4bff-8a68-8226d333a138.png)

# Оценка точности модели
Была проведена оценка точности моделей. Полученные результаты свидетельствуют о том, что наиболее точной является модель InceptionV3, а наименее точной - AlexNet

![image](https://user-images.githubusercontent.com/79449892/213930631-05698829-e07a-42dc-8271-b156f1b242ee.png)


# Источники

https://pytorch.org/docs/stable/torch.html

https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html

https://pytorch.org/vision/main/models/alexnet.html

https://pytorch.org/vision/main/models/inception.html
