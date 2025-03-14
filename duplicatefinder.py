'''Функции, позволяющие сравнивать изображения на схожесть.
Алгоритм сравнения основан на анализе средней интенсивности изображения и его четвертей.

intensities() переводит изображение в вектор интенсивностей, который и используется для сравнения
similarity() указывает уровень близости между двумя векторами
similarity_images() указывает уровень близости между двумя изображениями

Также можно пользоваться более простыми готовыми функциями
is_similar_images() и is_similar()
которые вернут булево значение "похоже/не похоже" для двух изображений или векторов соответственно
'''
from numbers import Real
from pathlib import Path
from typing import BinaryIO, Iterable
import cv2
from cv2.typing import MatLike
import numpy as np
import numpy.typing as npt
import PIL.Image as pil

num_dtype = np.float32
Vector = npt.NDArray[num_dtype]

# Коэффициенты яркости Y преобразования sRGB -> xyY для компонент BGR (ITU-R BT.709)
BGR_COEFFS = np.array([0.072186, 0.715158, 0.212656], dtype=num_dtype)

def open_image(image_path: str|Path|BinaryIO) -> MatLike:
    """Функция, открывающая изображение по его пути, возвращается матрица пикселей в BGR
    image_path: путь к изображению
    """
    # Существует проблема с OpenCV, не позволяющая работать с файлами вне рабочей директории
    #image = cv2.imread(image_path,cv2.IMREAD_COLOR)
    with pil.open(image_path) as img:
        is_grayscale = img.mode == 'L'
        if is_grayscale:
            # Прямой доступ к буферу данных, сразу float32
            img = img.convert('F')
        elif not is_grayscale and img.mode != 'RGB':
            img = img.convert('RGB')
        image = np.array(img, dtype=num_dtype, copy=False)

    if not is_grayscale:
        # RGB -> BGR
        image = image[..., ::-1]

    return image

def intensities(
    image_path: str|Path|BinaryIO,
    partition_level: int = 2,
    blur_ksize: int|tuple[int, int] = (3, 3)
) -> Vector:
    '''
    https://gist.github.com/liamwhite/b023cdba4738e911293a8c610b98f987
    Алгоритм основан на анализе средней интенсивности изображения и его четвертей

    image_path: путь к изображению

    partition_level устанавливает размерность вектора, которая будет равна
    sum{n=1,partition_level}(n^2)
    В принципе, можно получать и одномерные вектора, но меньшая размерность приводит к тому,
    что вектора чаще будут считаться похожими

    blur_ksize: Размер ядра размытия Гаусса, (0,0) для отключения

    Возвращается вектор, соответствующий изображению, полученному на входе
    '''
    image = open_image(image_path)

    # Размытие изображения
    if isinstance(blur_ksize, int):
        blur_ksize = (blur_ksize, blur_ksize)
    if not all(blur_ksize):
        image = cv2.GaussianBlur(image, blur_ksize, 0)

    # Приведение к значениям яркости (градациям серого)
    if image.ndim != 2:
        image = np.dot(image, BGR_COEFFS)

    # Предвычисление интегрального изображения для яркости
    integral = cv2.integral(image)

    # Выделяем память для вектора, характеризующего изображение
    features = np.empty(sum(level**2 for level in range(1, partition_level+1)), dtype=num_dtype)
    features_idx = 0

    # Высота и ширина всего изображения
    y_size, x_size = image.shape[:2]

    for current_partition_level in range(1, partition_level + 1):
        # Высота и ширина текущего разбиения
        y_step = y_size / current_partition_level
        x_step = x_size / current_partition_level

        for (y0, y1), (x0, x1) in generate_grid_coords(
            x_size,
            y_size,
            current_partition_level,
            x_step=x_step,
            y_step=y_step
        ):
            # Площадь части
            area = (y1 - y0) * (x1 - x0)

            # Расчёт компоненты итогового вектора
            if area == 0:
                features[features_idx] = 0.0
            else:
                # Быстрый расчёт суммы через интегральное изображение
                # и получение средней интенсивности в области
                features[features_idx] = np.divide(
                    _get_area_integral_sum(
                        integral, ((x0, x1), (y0, y1))
                    ),
                    area
                )

            features_idx +=1

    # Вектор, характеризующий изображение
    return features

def _get_area_integral_sum(integral: MatLike, coord: tuple[tuple[int, int], tuple[int, int]]) -> MatLike:
    (x0, x1), (y0, y1) = coord
    return integral[y1, x1] - integral[y1, x0] - integral[y0, x1] + integral[y0, x0]

def generate_grid_coords(
    x_size: int,
    y_size: int,
    side_partitions_num: int = 2,
    *,
    x_step: float|None = None,
    y_step: float|None = None
):
    '''
    Генерирует координаты регионов для разбиения изображения на side_partitions_num^2 частей

    Args:
        x_size: ширина изображения
        y_size: высота изображения
        side_partitions_num: Количество частей по каждой оси

    Yields:
        Пары ((y0, y1), (x0, x1)) - координаты регионов
    '''
    # Чистые размеры части без округлений
    x_step = x_step or (x_size / side_partitions_num)
    y_step = y_step or (y_size / side_partitions_num)

    x_points = np.floor(np.arange(0, x_size+1, x_step)).astype(int)
    x_starts = x_points[:-1]
    x_ends = x_points[1:]
    y_points = np.floor(np.arange(0, y_size+1, y_step)).astype(int)
    y_starts = y_points[:-1]
    y_ends = y_points[1:]

    # Разбиение слева направо
    x_parts = list(zip(x_starts, x_ends))
    # Разбиение сверху вниз
    y_parts = zip(y_starts, y_ends)

    for y0, y1 in y_parts:
        for x0, x1 in x_parts:
            yield ((y0, y1), (x0, x1))

def division_on_partitions_iter(image: MatLike, side_partitions_num: int = 2):
    '''Разбиение изображения равномерной сеткой на side_partitions_num^2 частей
    '''
    # Чистые размеры части без округлений
    partition_x_size_raw = image.shape[1] / side_partitions_num
    partition_y_size_raw = image.shape[0] / side_partitions_num

    for y, x in generate_grid_coords(
        image.shape[1],
        image.shape[0],
        side_partitions_num,
        x_step=partition_x_size_raw,
        y_step=partition_y_size_raw
    ):
        yield image[y[0]:y[1], x[0]:x[1]]

def get_rectangles_iter(image: MatLike, partition_level: int = 2):
    '''Последовательное всё более дробное разбиение изображения на части и получение этих частей.
    Разбиение идёт от возврата самого изображения до сетки со стороной partition_level частей
    '''
    for current_partition_level in range(1, partition_level + 1):
        yield from division_on_partitions_iter(
            image,
            side_partitions_num=current_partition_level
        )

def rect_sum(rect: MatLike) -> np.floating:
    '''Координата вектора — средняя относительная яркость
    '''
    sums: npt.NDArray[np.generic]|np.generic
    if rect.ndim > 2 and rect.shape[2]>2:
        sums = np.sum(rect, axis=(0, 1))
        if rect.shape[2]>3:
            sums = sums[-3:] + sums[3]
        # При использовании cv2.imread вместо plt.imread
        #sums: npt.NDArray[np.floating] = sums[:3]
    else:
        sums = np.sum(rect)
    intensity = (sums / (rect.shape[0] * rect.shape[1])) * BGR_COEFFS
    avg_intensity: np.floating = round(intensity.mean(), 6)

    return avg_intensity

def intensities_iter(
    image_list: Iterable[str|Path|BinaryIO],
    **kwargs
):
    '''Итератор, возвращающий вектора для набора полученных изображений
    '''
    for image in image_list:
        yield intensities(image, **kwargs)

def similarity(vector1: Vector|tuple|list, vector2: Vector|tuple|list) -> np.floating:
    '''Уровень похожести между векторами.
    0 - идентичные, 1 - максимально разные
    '''
    # Преобразование необходимо для вычисления разницы векторов
    vector1 = np.asarray(vector1, dtype=num_dtype) if not isinstance(vector1, np.ndarray) else vector1
    vector2 = np.asarray(vector2, dtype=num_dtype) if not isinstance(vector2, np.ndarray) else vector2

    if np.array_equal(vector1, vector2):
        return np.floating(0)

    norm_sum = np.sqrt(np.dot(vector1, vector1)) + np.sqrt(np.dot(vector2, vector2))

    if np.isclose(norm_sum, 0):
        # Неопределённость 0/0, но так как distance=0, то идентичны
        return np.floating(0)

    diff = vector1 - vector2
    distance = np.sqrt(np.dot(diff, diff))

    return distance/norm_sum

def is_similar(vector1: Vector, vector2: Vector, threshold: Real=0.1) -> bool:
    '''Похожи ли два вектора
    threshold: уровень разницы, больше которого вектора считаются разными
    Возвращается bool
    '''
    return similarity(vector1,vector2) < threshold

def similarity_images(
        image1: str|Path|BinaryIO,
        image2: str|Path|BinaryIO,
        **kwargs
    ) -> np.floating:
    '''Уровень похожести между изображениями.
    0 - идентичные, 1 - максимально разные
    '''
    return similarity(intensities(image1, **kwargs), intensities(image2, **kwargs))

def is_similar_images(
        image1: str|Path|BinaryIO,
        image2: str|Path|BinaryIO,
        threshold: Real = 0.1,
        **kwargs
    ) -> bool:
    '''Похожи ли два изобоажения
    threshold: уровень разницы, больше которого изображения считаются разными
    Возвращается bool
    '''
    return similarity_images(image1, image2, **kwargs) < threshold
