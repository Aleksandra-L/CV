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


# Выводы по работе


# Источники
https://habr.com/ru/company/joom/blog/445354/
https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html
https://docs.opencv.org/3.4/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dab65c042ed62c9e9e095a1e7e41fe2773
https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
https://habr.com/ru/post/106302/
https://ru.wikipedia.org/wiki/Масштабно-инвариантная_трансформация_признаков
https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
