# Теоретическая часть
Была разработана система детектирования объектов двумя способами:
- поиск шаблона на изображении (Template matching)
- поиск ключевых точек на изображении (SIFT)
Алгоритм Template matching представляет собой поиск на исходном изображении места, наиболее похожего на шаблон. Способ нахождения этого места зависит от выбранного метода. В данной работе использовался метод SQDIFF. Он представляет собой поиск разницы евклидова расстояния между значениями пикселями. Из найденных расстояний выбирается наименьшее - расположение этого значения представляет собой верхнюю левую точку найденной части изображения. Формула:

![image](https://user-images.githubusercontent.com/79449892/213917578-cb854213-10fc-46ca-9760-5661e129c933.png)


Алгоритм SIFT (также известный как масштабно-инвариантная трансформация признаков) представляет собой поиск ключевых точек изображения. Для каждого из изображений находятся ключевые точки и их дескрипторы, после чего они сравниваются между собой. Преимуществом алгоритма является инвариативность по отношению к масштабированию, ориентации изображения, изменению освещенности и частично к афинными преобразованиям. Нахождение ключевых точек осуществляется при помощи построения пирамиды гауссианов и поиска разницы гауссианов. В графическом виде схема может быть представлена следующим образом:

![image](https://user-images.githubusercontent.com/79449892/213917992-4803d0e1-2801-4f78-827b-0d67dde0a372.png)

# Описание разработанной системы
При реализации поиска по шаблону была использована функция matchTemplate() OpenCV. Для полученного значения был осуществлен поиск наименьшего значения и его расположения, которое было принято за верхнюю левую точку найденной области.

Функция поиска по шаблону, реализованная в системе:
```
def template_match(image_og, image_temp):
    w, h = image_temp.shape[::-1]
    res = cv2.matchTemplate(image_og, image_temp,cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(image_og,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()
    cv2.rectangle(image_og,top_left, bottom_right, 255, 2)
    cv2.imshow('image', image_og)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```
При реализации алгоритма SIFT было использовано два способа поиска Brute-force matcher и FLANN based matcher. Brute-force matcher сранивает дескрипторы каждой ключевой точки изображения со всеми дескрипторами ключевых точек второго изображения и вычисляет расстояние между ними (через евклидово расстояние). Возвращаются ближайшие значения.

```
def sift_bf(image_og, image_temp):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image_og, None)
    kp2, des2 = sift.detectAndCompute(image_temp, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(image_og, kp1, image_temp, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
```

FLANN based matcher включает в себя несколько алгоритмов поиска, оптимизированных для быстрого поиска ближайших соседей., и работает значительно быстрее для объемных датасетов.

```
def sift_flann(image_og, image_temp):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image_og, None)
    kp2, des2 = sift.detectAndCompute(image_temp, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params,search_params)
    knn_matches = matcher.knnMatch(des1, des2, 2)

    ratio_thresh = 0.6
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    img_matches = np.empty((max(image_og.shape[0], image_temp.shape[0]), image_og.shape[1] + image_temp.shape[1], 3),
                           dtype=np.uint8)
    img3 = cv2.drawMatches(image_og, kp1, image_temp, kp2, good_matches[:10], img_matches,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
```
# Результаты работы системы
Исходное изображения, по которому осуществлялся поиск:

![Field](https://user-images.githubusercontent.com/79449892/213919597-583396f7-f0b8-43bf-bcc1-8a5eecf9e852.jpg)

Часть исходного изображения, поиск которой осуществлялся:

![Field_test](https://user-images.githubusercontent.com/79449892/213919604-4f8bb645-117c-4e99-8e9f-347eec9e9ad9.jpg)

Результат поиска template matching:

![image](https://user-images.githubusercontent.com/79449892/213919483-d13e3b76-093b-429e-b13b-859c9a17d6aa.png)

Резльтат поиска ключевых точек BFMatcher (10 лучших значений):

![image](https://user-images.githubusercontent.com/79449892/213919552-38cfd04a-e2c8-4769-8ea1-7d5eba4ceaf6.png)

Резльтат поиска ключевых точек FLANN (10 лучших значений):

![image](https://user-images.githubusercontent.com/79449892/213919571-fc9366e5-b960-4a6d-9a60-7e2906fc372d.png)

Было проведено тестирование работы системы на подборке изображений архитектуры. Результаты представлены ниже.

Исходное изображение:

![trevi](https://user-images.githubusercontent.com/79449892/213924153-bdbc228f-ee75-41ec-9150-da9d4ec3ed1d.jpg)

Изображение для поиска:

![image](https://user-images.githubusercontent.com/79449892/213924271-7920c5b6-5533-4751-b2e5-263b5db35022.png)

Template matching:

![image](https://user-images.githubusercontent.com/79449892/213924131-49640564-2543-4c0c-9152-f18af31a6f75.png)

BFMatcher:

![image](https://user-images.githubusercontent.com/79449892/213924199-0918d0d5-b589-4488-a13c-d7e903a0441f.png)

FLANN matcher:

![image](https://user-images.githubusercontent.com/79449892/213924210-2c4e7546-e8b2-48a8-b95f-637ca50a8d99.png)

Изображение для поиска:

![trevi4](https://user-images.githubusercontent.com/79449892/213924340-a06aa3d7-cc70-4ff7-8dea-e48dbbc87e8c.jpg)

Template matching:

![image](https://user-images.githubusercontent.com/79449892/213924382-6581d8bb-ba64-4a25-82d5-4ddd9ba2922a.png)

BFMatcher:

![image](https://user-images.githubusercontent.com/79449892/213924416-cc446fe9-f2d4-49d1-832f-84abdf1a1ee3.png)

FLANN matcher:

![image](https://user-images.githubusercontent.com/79449892/213924431-911d7c8d-eb59-491b-b72d-188e421cce1d.png)

Изображение для поиска:

![trevi3](https://user-images.githubusercontent.com/79449892/213924437-dc42cce0-799a-4a3c-83c5-49b32386458e.jpg)

Template matching:

![image](https://user-images.githubusercontent.com/79449892/213924460-48a12abd-26bb-4ca9-84fa-cefa253ecb63.png)

BFMatcher:

![image](https://user-images.githubusercontent.com/79449892/213924483-9bc7d6eb-05e2-42b2-a69d-cd566cf4b8d4.png)

FLANN matcher:

![image](https://user-images.githubusercontent.com/79449892/213924495-42955956-0f74-449d-ab59-285868180cea.png)

Изображение для поиска:

![trevi2](https://user-images.githubusercontent.com/79449892/213924525-a8a34a6e-3f91-45c1-8a59-df19656c1b3d.jpg)

Template matching:

![image](https://user-images.githubusercontent.com/79449892/213924521-fc20030f-ad6a-48db-bd10-f05c22b1640b.png)

BFMatcher:

![image](https://user-images.githubusercontent.com/79449892/213924553-b20a4877-78b9-4e4e-81e1-4ec6d56e2fa7.png)

FLANN matcher:

![image](https://user-images.githubusercontent.com/79449892/213924566-65ac4b5f-2a8d-46ba-9d03-df5c5b22c629.png)

Изображение для поиска:

![trevi1](https://user-images.githubusercontent.com/79449892/213924577-ef76f90f-a42d-4e18-9d2d-ffe02a9149ea.jpg)

Template matching:

![image](https://user-images.githubusercontent.com/79449892/213924630-e1fcec4e-ac4c-4527-b723-2da661461e61.png)

BFMatcher:

![image](https://user-images.githubusercontent.com/79449892/213924660-be5be6eb-73af-43ba-a26f-b54a553b433f.png)

FLANN matcher:

![image](https://user-images.githubusercontent.com/79449892/213924669-e90096d2-6295-4cc4-8146-e36e757259f6.png)

Исходное изображение:

![isaac](https://user-images.githubusercontent.com/79449892/213924767-a9089f86-e2fc-4d31-a44d-6a3fd1910bf5.jpg)

Изображение для поиска:

![isaac2](https://user-images.githubusercontent.com/79449892/213924776-f9784384-6eb5-4cbc-887d-7cdd117e97cc.jpg)

Template matching:

![image](https://user-images.githubusercontent.com/79449892/213924722-2bfd40aa-23a4-4364-9d1a-c308068d490a.png)

BFMatcher:

![image](https://user-images.githubusercontent.com/79449892/213924747-f6761feb-ffeb-42f3-a262-852331f7b57a.png)

FLANN matcher:

![image](https://user-images.githubusercontent.com/79449892/213924764-3916fe4c-4824-4e7f-a5be-a8cd6631cb49.png)

Изображение для поиска:

![isaac3](https://user-images.githubusercontent.com/79449892/213924795-d8c7e07a-5505-4387-a2ba-0defaae13eef.jpg)

Template matching:

![image](https://user-images.githubusercontent.com/79449892/213924842-6542fe6a-19c2-435d-b6d0-f34327eec47d.png)

BFMatcher:

![image](https://user-images.githubusercontent.com/79449892/213924813-ed700f4c-0131-4e8d-8603-014dd15f49c1.png)

FLANN matcher:

![image](https://user-images.githubusercontent.com/79449892/213924824-e26cae20-99e4-45e4-a776-a0771a22e7af.png)

Исходное изображение:

![Eifel](https://user-images.githubusercontent.com/79449892/213924862-e77d7e76-67e4-46cd-8b8e-9ab7ce1c04eb.jpg)

Изображение для поиска:

![Eifel2](https://user-images.githubusercontent.com/79449892/213924871-473b4ac4-4885-4ba8-a964-ce2fd56f586b.jpg)

Template matching:

![image](https://user-images.githubusercontent.com/79449892/213924899-9432e8fd-21b4-4fa9-9fe3-5e6cc0cd943f.png)

BFMatcher:

![image](https://user-images.githubusercontent.com/79449892/213924914-72ac3b25-ec93-41f0-8a38-281f45a31424.png)

FLANN matcher:

![image](https://user-images.githubusercontent.com/79449892/213924925-2c1bfb34-dc5c-4736-ba59-15e26815c605.png)

Изображение для поиска:

![Eifel3](https://user-images.githubusercontent.com/79449892/213924989-a00183cf-eaa5-48b5-9969-6c42507cb1ab.jpg)

Template matching:

![image](https://user-images.githubusercontent.com/79449892/213924980-ff991c8b-5bcd-448b-b84f-4e893c894b5f.png)

BFMatcher:

![image](https://user-images.githubusercontent.com/79449892/213924949-8bd69423-b2f9-40d9-aaa5-a73b5db41e82.png)

FLANN matcher:

![image](https://user-images.githubusercontent.com/79449892/213924968-f5479825-d692-416f-8c11-8f7b1c46b870.png)

Изображение для поиска:

![Eifel4](https://user-images.githubusercontent.com/79449892/213925003-e1305b52-e3d2-47c2-a8c2-e8053b324502.jpg)

Template matching:

![image](https://user-images.githubusercontent.com/79449892/213925036-97204906-bd5f-4f57-88fd-74991f251e19.png)

BFMatcher:

![image](https://user-images.githubusercontent.com/79449892/213925075-91cb9d9b-aaa4-4bad-94fd-8b06cae6fcdc.png)

FLANN matcher:

![image](https://user-images.githubusercontent.com/79449892/213925080-4fd85078-d089-4223-a24f-32c14fde06e9.png)

# Выводы по работе
В результате выполнения лабораторной работы была создана система детектирования объектов. После тестирования системы на различных изображениях можно сделать вывод, что поиск по шаблону абсолютно неэффективен, если изображение не является частью исходного. Алгоритм SIFT довольно точен, несмотря на изменения ракурса и освещения, однако иногда и он не справляется с поиском. Поиск идет не очень точно на изображениях с большим количеством деталей.

# Источники
https://habr.com/ru/company/joom/blog/445354/

https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html

https://docs.opencv.org/3.4/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dab65c042ed62c9e9e095a1e7e41fe2773

https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

https://habr.com/ru/post/106302/

https://ru.wikipedia.org/wiki/Масштабно-инвариантная_трансформация_признаков

https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
