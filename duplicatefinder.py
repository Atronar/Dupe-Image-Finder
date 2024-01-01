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
import matplotlib.pyplot as plt

Vector = tuple[float, float, float, float, float]

def intensities(image_path: str|Path|BinaryIO) -> Vector:
    '''
    https://gist.github.com/liamwhite/b023cdba4738e911293a8c610b98f987
    Алгоритм основан на анализе средней интенсивности изображения и его четвертей

    image_path: путь к изображению

    Возвращается вектор, соответствующий изображению, полученному на входе
    В принципе, можно получать и одномерные вектора, но меньшая размерность приводит к тому,
    что вектора чаще будут считаться похожими
    '''
    # Существует проблема с OpenCV, не позволяющая работать с файлами вне рабочей директории
    image = plt.imread(image_path, 0)
    image = image[..., ::-1]
    #image = cv2.imread(image_path,cv2.IMREAD_COLOR)
    image = cv2.GaussianBlur(image, (3,3), 0)

    half_x = round(image.shape[1] / 2)
    half_y = round(image.shape[0] / 2)

    nw_rect = image[0:half_y, 0:half_x]
    ne_rect = image[0:half_y, (half_x + 1):image.shape[1]]
    sw_rect = image[(half_y + 1):image.shape[0], 0:half_x]
    se_rect = image[(half_y + 1):image.shape[0], (half_x + 1):image.shape[1]]

    average_intensity = rect_sum(image)
    nw_intensity = rect_sum(nw_rect)
    ne_intensity = rect_sum(ne_rect)
    sw_intensity = rect_sum(sw_rect)
    se_intensity = rect_sum(se_rect)

    # Вектор, характеризующий изображение
    return average_intensity, nw_intensity, ne_intensity, sw_intensity, se_intensity

def rect_sum(rect: MatLike) -> np.floating:
    '''Координата вектора — средняя относительная яркость
    '''
    # Коэффициенты яркости Y преобразования sRGB -> xyY для компонент RGB
    coefficients = np.array([0.212656, 0.715158, 0.072186], np.float32)

    sums: np.floating = np.sum(rect, axis=(0, 1))
    intensity = (sums / (rect.shape[0] * rect.shape[1])) * coefficients
    avg_intensity: np.floating = round(intensity.mean(), 6)

    return avg_intensity

def intensities_iter(image_list: Iterable[str|Path|BinaryIO]):
    '''Итератор, возвращающий вектора для набора полученных изображений
    '''
    for image in image_list:
        yield intensities(image)

def similarity(vector1: Vector, vector2: Vector) -> np.floating:
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
