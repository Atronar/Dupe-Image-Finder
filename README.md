Dupe Image Finder
==============
**Dupe Image Finder** — библиотека для нахождения дубликатов изображений

Для работы необходимо:
* [python 3.6](https://www.python.org/downloads/)
* [opencv](https://github.com/skvark/opencv-python/)

Методы:
------
[intensities(image_path)](#intensities)
[intensities_iter(image_list)](#intensities_iter)
[similarity(vector1, vector2)](#similarity)
[is_similar(vector1,vector2,threshold=0.1)](#is_similar)
[similarity_images(image1, image2)](#similarity_images)
[is_similar_images(image1,image2,threshold=0.1)](#is_similar_images)

#### intensities
-----
**intensities(str image_path) -> tuple(avg_intensivity, nw_intensivity, ne_intensivity, sw_intensivity, se_intensivity)**
Главная функция всего модуля.
Алгоритм основан на анализе средней интенсивности изображения и его четвертей ([алгоритм соответствует этому коду на Ruby](https://gist.github.com/liamwhite/b023cdba4738e911293a8c610b98f987))
Данная функция возвращает пятимерный вектор, соответствующий изображению, полученному на входе

#### intensities_iter
-----
**intensities_iter(iterable image_list) -> yield tuple**
Обёртка над [intensities](#intensities) для работы со списком файлов, функция работает как итератор.
Подразумевается, что с помощью данной функции будет создаваться БД, ставящая значения векторов в соответствие файлам, а в дальнейшем поиск дубликатов будет проводиться по этой уже заготовленной БД.

#### similarity
-----
**similarity(tuple vector1, tuple vector2) -> float ∈ [0; 1]**
Уровень похожести между двумя векторами, характеризуется значениями от 0 до 1.
0 — вектора идентичны
1 — вектора максимально разные

#### is_similar
-----
**is_similar(tuple vector1, tuple vector2, float threshold=0.1) -> bool**
Функция, возвращающая *True*, если вектора похожи, иначе возвращается *False*.
Параметр *threshold* принимает значения от 0 до 1 и устанавливает порог, по которому определяется, похожи ли вектора.

#### similarity_images
-----
**similarity_images(image1, image2) -> float ∈ [0; 1]**
Делает то же, что и [similarity](#similarity), но принимает на вход изображения.

#### is_similar_images
------
**is_similar_images(image1,image2,threshold=0.1) -> bool**
Делает то же, что и [is_similar](#is_similar), но принимает на вход изображения.