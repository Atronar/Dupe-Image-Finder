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

def intensities(image_path: str|Path|BinaryIO, partition_level: int = 2) -> Vector:
    '''
    https://gist.github.com/liamwhite/b023cdba4738e911293a8c610b98f987
    Алгоритм основан на анализе средней интенсивности изображения и его четвертей

    image_path: путь к изображению

    Возвращается вектор, соответствующий изображению, полученному на входе
    
    partition_level устанавливает размерность вектора, которая будет равна
    sum{n=1,partition_level}(n^2)
    В принципе, можно получать и одномерные вектора, но меньшая размерность приводит к тому,
    что вектора чаще будут считаться похожими
    '''
    # Существует проблема с OpenCV, не позволяющая работать с файлами вне рабочей директории
    image = np.array(pil.open(image_path))
    image = image[..., ::-1]
    #image = cv2.imread(image_path,cv2.IMREAD_COLOR)
    image = cv2.GaussianBlur(image, (3,3), 0)

    rectangles = get_rectangles_iter(image, partition_level)

    intensities_list = map(rect_sum, rectangles)

    # Вектор, характеризующий изображение
    return np.fromiter(intensities_list, dtype=num_dtype)

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

    x_starts = np.floor(np.arange(x_step * side_partitions_num)).astype(int)
    x_ends = np.ceil(np.arange(x_step * (side_partitions_num) + 1)).astype(int)
    y_starts = np.floor(np.arange(y_step * side_partitions_num)).astype(int)
    y_ends = np.ceil(np.arange(y_step * (side_partitions_num) + 1)).astype(int)

    # Разбиение слева направо
    x_parts = zip(x_starts, x_ends)
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
    for current_partition_level in range(partition_level):
        yield from division_on_partitions_iter(
            image,
            side_partitions_num=current_partition_level + 1
        )

def rect_sum(rect: MatLike) -> np.floating:
    '''Координата вектора — средняя относительная яркость
    '''
    sums: npt.NDArray[np.generic]|np.generic = np.sum(rect, axis=(0, 1))
    if hasattr(sums, '__len__') and len(sums)>3:
        sums = sums[-3:]
        # При использовании cv2.imread вместо plt.imread
        #sums: npt.NDArray[np.floating] = sums[:3]
    intensity = (sums / (rect.shape[0] * rect.shape[1])) * BGR_COEFFS
    avg_intensity: np.floating = round(intensity.mean(), 6)

    return avg_intensity

def intensities_iter(image_list: Iterable[str|Path|BinaryIO]):
    '''Итератор, возвращающий вектора для набора полученных изображений
    '''
    for image in image_list:
        yield intensities(image)

def similarity(vector1: Vector|tuple|list, vector2: Vector|tuple|list) -> np.floating:
    '''Уровень похожести между векторами.
    0 - идентичные, 1 - максимально разные
    '''
    # Преобразование необходимо для вычисления разницы векторов
    vector1 = np.array(vector1) if not isinstance(vector1, np.ndarray) else vector1
    vector2 = np.array(vector2) if not isinstance(vector2, np.ndarray) else vector2

    distance = np.linalg.norm(vector1 - vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    return distance/(norm1+norm2)

def is_similar(vector1: Vector, vector2: Vector, threshold: Real=0.1) -> bool:
    '''Похожи ли два вектора
    threshold: уровень разницы, больше которого вектора считаются разными
    Возвращается bool
    '''
    return similarity(vector1,vector2) < threshold

def similarity_images(
        image1: str|Path|BinaryIO,
        image2: str|Path|BinaryIO
    ) -> np.floating:
    '''Уровень похожести между изображениями.
    0 - идентичные, 1 - максимально разные
    '''
    return similarity(intensities(image1), intensities(image2))

def is_similar_images(
        image1: str|Path|BinaryIO,
        image2: str|Path|BinaryIO,
        threshold: Real = 0.1
    ) -> bool:
    '''Похожи ли два изобоажения
    threshold: уровень разницы, больше которого изображения считаются разными
    Возвращается bool
    '''
    return similarity_images(image1, image2) < threshold
