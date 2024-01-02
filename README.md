Dupe Image Finder
==============
**Dupe Image Finder** — библиотека для нахождения дубликатов изображений

Для работы необходимо:
* [python 3.6](https://www.python.org/downloads/)
* [opencv](https://github.com/skvark/opencv-python/)

Методы:
------
[intensities(image_path)](#intensities)<br>
[intensities_iter(image_list)](#intensities_iter)<br>
[similarity(vector1, vector2)](#similarity)<br>
[is_similar(vector1,vector2,threshold=0.1)](#is_similar)<br>
[similarity_images(image1, image2)](#similarity_images)<br>
[is_similar_images(image1,image2,threshold=0.1)](#is_similar_images)<br>

#### intensities
-----
**intensities(image_path: str|Path|BinaryIO) -> array**<br>
Главная функция всего модуля.<br>
Алгоритм основан на анализе средней интенсивности изображения и его четвертей ([алгоритм соответствует этому коду на Ruby](https://gist.github.com/liamwhite/b023cdba4738e911293a8c610b98f987))<br>
Данная функция возвращает пятимерный вектор, соответствующий изображению, полученному на входе

#### intensities_iter
-----
**intensities_iter(image_list: Iterable[str|Path|BinaryIO]) -> yield array**<br>
Обёртка над [intensities](#intensities) для работы со списком файлов, функция работает как итератор.<br>
Подразумевается, что с помощью данной функции будет создаваться БД, ставящая значения векторов в соответствие файлам, а в дальнейшем поиск дубликатов будет проводиться по этой уже заготовленной БД.

#### similarity
-----
**similarity(vector1: array, vector2: array) -> float ∈ [0; 1]**<br>
Уровень похожести между двумя векторами, характеризуется значениями от 0 до 1.<br>
0 — вектора идентичны<br>
1 — вектора максимально разные

#### is_similar
-----
**is_similar(vector1: array, vector2: array, threshold: float=0.1) -> bool**<br>
Функция, возвращающая *True*, если вектора похожи, иначе возвращается *False*.<br>
Параметр *threshold* принимает значения от 0 до 1 и устанавливает порог, по которому определяется, похожи ли вектора.

#### similarity_images
-----
**similarity_images(image1: str|Path|BinaryIO, image2: str|Path|BinaryIO) -> float ∈ [0; 1]**<br>
Делает то же, что и [similarity](#similarity), но принимает на вход изображения.

#### is_similar_images
------
**is_similar_images(image1: str|Path|BinaryIO, image2: str|Path|BinaryIO, threshold: float=0.1) -> bool**<br>
Делает то же, что и [is_similar](#is_similar), но принимает на вход изображения.
