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
from io import BufferedIOBase
try:
    import cv2
    cv2_use = True
except ImportError:
    cv2_use = False
from cv2.typing import MatLike
import numpy as np
import numpy.typing as npt
try:
    import PIL.Image as pil
    pil_use = True
except ImportError:
    pil_use = False
try:
    import pyvips
    pyvips_use = True
except ImportError:
    pyvips_use = False

num_dtype = np.float32
Vector = npt.NDArray[num_dtype]

# Коэффициенты яркости Y преобразования sRGB -> xyY для компонент BGR (ITU-R BT.709)
BGR_COEFFS = np.array([0.072186, 0.715158, 0.212656], dtype=num_dtype)

def open_image(image_path: str|Path|BinaryIO) -> MatLike:
    """Функция, открывающая изображение по его пути, возвращается матрица пикселей в BGR
    image_path: путь к изображению
    """
    try:
        if cv2_use:
            return open_image_cv(image_path)
    except:
        gif = True
        pass
    if pyvips_use:
        return open_image_vips(image_path)
    if pil_use:
        return open_image_pil(image_path)
    if gif:
        raise ModuleNotFoundError("Для работы с данным типом изображений необходимо установить хотя бы одну из этих библиотек: pyvips, pil")
    else:
        raise ModuleNotFoundError("Необходимо установить хотя бы одну из этих библиотек: pyvips, pil, opencv")

def open_image_vips(image_path: str|Path|BinaryIO, file_format: str|None=None) -> npt.NDArray[num_dtype]:
    """Открытие изображения в BGR с помощью libvips

    image_path: путь к изображению
    file_format: (необязательно) формат входного файла

    Возвращается ndarray[float32] — матрица пикселей в BGR
    """
    if not pyvips_use:
        raise ModuleNotFoundError("pyvips не установлен. Необходимо использовать функцию open_image()")

    if isinstance(image_path, BinaryIO):
        # Создать источник из буфера
        pos = image_path.tell()
        try:
            image_data = image_path.read()
        finally:
            image_path.seek(pos) # возвращаем указатель файла к исходной позиции
        if file_format:
            img = pyvips.Image.new_from_buffer(image_data, "", format=file_format)
        else:
            source = pyvips.Source.new_from_memory(image_data)
            img = pyvips.Image.new_from_source(source, "")  # автоопределение формата
        del image_data
    else:
        img = pyvips.Image.new_from_file(image_path, access="sequential")

    if not img:
        raise FileNotFoundError("Невозможно прочитать файл")

    if isinstance(img, list):
        if not img:
            raise FileNotFoundError("Невозможно прочитать файл")
        image: pyvips.Image = img[0]
    else:
        image: pyvips.Image = img
    del img

    # удаляем альфа-канал, заменяя его белым цветом
    if image.hasalpha():
        image = image.flatten(background=255)

    # Обработка Grayscale или одноканальных данных
    if image.bands == 1 or image.interpretation in ["b-w", "greyscale"]:
        return image.numpy(dtype=num_dtype).squeeze()  # удаление оси каналов (H, W)
    # Конвертация CMYK → RGB
    elif image.interpretation == "cmyk":
        # Используем встроенный профиль sRGB
        image = image.icc_transform("srgb")
    # Обработка других не-RGB случаев (например, многоканальные не-RGB)
    elif "rgb" not in image.interpretation:
        # Конвертация в RGB через промежуточное пространство
        image = image.colourspace("srgb")
    # Конвертация RGB → BGR
    return image[2::-1].numpy(dtype=num_dtype)

def open_image_cv(image_path: str|Path|BinaryIO) -> MatLike:
    """Открытие изображения в BGR с помощью opencv

    image_path: путь к изображению

    Возвращается ndarray[float32] — матрица пикселей в BGR
    """
    if not cv2_use:
        raise ModuleNotFoundError("opencv не установлен. Необходимо использовать функцию open_image()")

    def load_image(image_path: str|Path|BinaryIO|BufferedIOBase, flags: int) -> MatLike:
        if isinstance(image_path, (BinaryIO, BufferedIOBase)):
            pos = image_path.tell()
            try:
                image_data = image_path.read()
            finally:
                image_path.seek(pos)
            np_array = np.frombuffer(image_data, dtype=np.uint8)
            img = cv2.imdecode(np_array, flags)
        else:
            image_path = Path(image_path).absolute()
            img = cv2.imread(str(image_path), flags)
            if img is None:
                with open(image_path, "rb") as image_file:
                    return load_image(image_file, flags)
        if img is None:
            raise FileNotFoundError("Невозможно прочитать файл")
        return img
    img = load_image(image_path, cv2.IMREAD_UNCHANGED)

    channels = 1 if img.ndim == 2 else img.shape[2]
    is_grayscale = channels < 3
    has_alpha = channels in (2, 4)

    # Изображения без альфа-канала возвращаем как просто BGR/Grayscale
    if not has_alpha:
        if not is_grayscale and np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 0], img[..., 2]):
            img = img[..., 0]
        return img.astype(num_dtype, copy=False)

    # Приведение фактического серого из RGBA в градации серого без потери альфа-канала
    # Сравнение значений каналов
    if not is_grayscale and np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 0], img[..., 2]):
        img = img[:, :, [0, 3]].view(np.uint8).reshape(*img.shape[:2], 2)
        is_grayscale = True

    # Удаление альфа-канала
    # Простая конвертация заменяет альфу на чёрный вместо белого, что недопустимо
    # Нормализация альфа-канала в [0, 1]
    alpha = img[..., -1].astype(num_dtype, copy=False)[..., np.newaxis] / 255.0
    color = img[..., :-1].astype(num_dtype, copy=False)
    if is_grayscale:
        white = 255
    else:
        white = np.array([255.0, 255.0, 255.0], dtype=num_dtype)
    image = color * alpha + white * (1 - alpha)
    return image.astype(num_dtype)

def open_image_pil(image_path: str|Path|BinaryIO) -> npt.NDArray[num_dtype]:
    """Открытие изображения в BGR с помощью PIL

    image_path: путь к изображению

    Возвращается ndarray[float32] — матрица пикселей в BGR
    """
    if not pil_use:
        raise ModuleNotFoundError("pillow не установлен. Необходимо использовать функцию open_image()")

    with pil.open(image_path) as img:
        convert_to = None
        mode = img.mode
        is_grayscale = mode in ['L', 'LA']
        has_alpha = mode in ['RGBA', 'LA']
        is_rgb = 'RGB' in mode
        # Изображения с палитрой и альфа
        if img.mode == 'P':
            mode = img.palette.mode
            if "transparency" in img.info or mode in ['RGBA', 'LA']:
                has_alpha = True
                if mode == 'LA':
                    convert_to = 'LA'
                    is_grayscale = True
                else:
                    convert_to = 'RGBA'
                    is_rgb = True
            elif mode == 'L':
                convert_to = 'L'
                is_grayscale = True
            elif mode == 'RGB':
                convert_to = 'RGB'
                is_rgb = True

        # Приведение фактического серого из RGB(A) в градации серого без потери альфа-канала
        # Сравнение значений каналов
        arr = np.asarray(img, num_dtype)
        if is_rgb and np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
            convert_to = 'LA' if has_alpha else 'L'
            is_grayscale = True
            is_rgb = False

        if convert_to:
            img = img.convert(convert_to)
            arr = np.asarray(img, num_dtype)

        # Удаление альфа-канала
        # Простая конвертация заменяет альфу на чёрный вместо белого, что недопустимо
        if has_alpha:
            #arr = np.asarray(img)
            alpha = (arr[:, :, -1] * (1/255))       # [0, 1]
            color = arr[:, :, :-1].squeeze()        # [0, 255]
            if is_grayscale:
                white = 255
            else:
                alpha = alpha[..., np.newaxis]
                white = np.array([255.0, 255.0, 255.0], dtype=num_dtype)
            image = color * alpha + white * (1 - alpha)
            return image.astype(num_dtype, copy=False)
        # Серый и RGB конвертировать не нужно
        if not (is_grayscale or is_rgb):
            img = img.convert('RGB')
        image = np.asarray(img, dtype=num_dtype)

    if not is_grayscale:
        # RGB -> BGR
        image = image[..., ::-1]

    return image

def gauss_blur(image: MatLike, blur_ksize: int|tuple[int, int] = (3, 3), sigma: float = 0) -> MatLike:
    """Размытие изображения по Гауссу

    image: массив изображения
    blur_ksize: размеры гауссова ядра, должно быть нечётным
    sigma: стандартное отклонение

    Возвращается массив размытого изображения
    """
    if cv2_use:
        return gauss_blur_cv(image, blur_ksize, sigma)
    if pyvips_use:
        return gauss_blur_vips(image, blur_ksize, sigma)
    return gauss_blur_custom(image, blur_ksize, sigma)

def gauss_blur_vips(image: MatLike, blur_ksize: int|tuple[int, int] = (3, 3), sigma: float = 0) -> MatLike:
    """Размытие изображения по Гауссу с помощью libvips

    image: массив изображения
    blur_ksize: размеры гауссова ядра, должно быть нечётным
    sigma: стандартное отклонение

    Возвращается массив размытого изображения
    """
    if not pyvips_use:
        raise ModuleNotFoundError("pyvips не установлен. Необходимо использовать функцию gauss_blur()")

    if isinstance(blur_ksize, tuple):
        blur_ksize = min(blur_ksize)
    if not blur_ksize:
        return image

    if np.isclose(sigma, 0):
        sigma = 0.15 * blur_ksize + 0.35
    min_ampl = np.exp(-(blur_ksize//2)**2 / (2 * sigma**2))

    blurred_image = pyvips.Image.new_from_array(image).gaussblur(sigma, min_ampl=min_ampl)
    return blurred_image.numpy(dtype=num_dtype)

def gauss_blur_cv(image: MatLike, blur_ksize: int|tuple[int, int] = (3, 3), sigma: float = 0) -> MatLike:
    """Размытие изображения по Гауссу с помощью opencv

    image: массив изображения
    blur_ksize: размеры гауссова ядра, должно быть нечётным
    sigma: стандартное отклонение

    Возвращается массив размытого изображения
    """
    if not cv2_use:
        raise ModuleNotFoundError("opencv не установлен. Необходимо использовать функцию gauss_blur()")

    if isinstance(blur_ksize, int):
        blur_ksize = (blur_ksize, blur_ksize)
    if not all(blur_ksize):
        return image
    return cv2.GaussianBlur(image, blur_ksize, sigma)

def gauss_blur_custom(image: MatLike, blur_ksize: int|tuple[int, int]=(3, 3), sigma: float=0) -> MatLike:
    """Размытие изображения по Гауссу с помощью чистого numpy

    image: массив изображения
    blur_ksize: размеры гауссова ядра, должно быть нечётным
    sigma: стандартное отклонение

    Возвращается массив размытого изображения
    """
    if isinstance(blur_ksize, int):
        blur_ksize = (blur_ksize, blur_ksize)

    if not all(blur_ksize):
        return image

    kx, ky = blur_ksize

    if np.isclose(sigma, 0):
        sigma_x = 0.15 * kx + 0.35
        sigma_y = 0.15 * ky + 0.35
    else:
        sigma_x = sigma_y = sigma

    def gaussian_kernel_1d(size: int, sigma: float) -> npt.NDArray:
        """1D ядро Гаусса."""
        kernel = np.exp(-(np.arange(size, dtype=num_dtype) - size//2)**2 / (2 * sigma**2))
        return kernel / kernel.sum(dtype=num_dtype) # Нормализация

    # Генерация разделяемых 1D ядер Гаусса
    kernel_x = gaussian_kernel_1d(kx, sigma_x)
    kernel_y = gaussian_kernel_1d(ky, sigma_y)

    pad_x = kx // 2
    pad_y = ky // 2
    if image.ndim > 2:
        padded_image = np.pad(image, ((pad_x, pad_x), (pad_y, pad_y), (0, 0)), mode='reflect')
    else:
        padded_image = np.pad(image, ((pad_x, pad_x), (pad_y, pad_y)), mode='reflect')

    # Векторизованная свертка
    blurred_x = np.apply_along_axis(
        lambda c: np.convolve(c, kernel_x, mode='valid'),
        1, padded_image
    )
    blurred = np.apply_along_axis(
        lambda r: np.convolve(r, kernel_y, mode='valid'),
        0, blurred_x
    )

    blurred = np.clip(blurred, 0, 255).astype(num_dtype, copy=False)
    return blurred

def integral_image(image: MatLike) -> MatLike:
    """
    Вычисляет интегральное изображение.

    image: массив изображения

    Возвращается массив интегрального изображения с нулевой рамкой — кумулятивная сумма по каждому измерению
    """
    if cv2_use:
        return integral_image_cv(image)
    return integral_image_custom(image)

def integral_image_cv(image: MatLike) -> MatLike:
    """
    Вычисляет интегральное изображение с помощью opencv.

    image: массив изображения

    Возвращается массив интегрального изображения с нулевой рамкой — кумулятивная сумма по каждому измерению
    """
    if not cv2_use:
        raise ModuleNotFoundError("opencv не установлен. Необходимо использовать функцию integral_image()")
    return cv2.integral(image)

def integral_image_custom(image: MatLike) -> MatLike:
    """
    Вычисляет интегральное изображение с помощью чистого numpy.

    image: массив изображения

    Возвращается массив интегрального изображения с нулевой рамкой — кумулятивная сумма по каждому измерению
    """
    # Выбор типа данных
    kind = image.dtype.kind
    dtype = (
        np.float32 if kind == 'f'
        else np.uint64 if kind == 'u'
        else np.int64
    )

    # Создаем нулевую рамку
    if image.ndim == 3:
        h, w, c = image.shape
        integral = np.zeros((h+1, w+1, c), dtype=dtype)
        integral[1:, 1:, :] = image.astype(dtype, copy=False).cumsum(0).cumsum(1)
    elif image.ndim == 2:
        h, w = image.shape
        integral = np.zeros((h+1, w+1), dtype=dtype)
        integral[1:, 1:] = image.astype(dtype, copy=False).cumsum(0).cumsum(1)
    elif image.ndim == 1:
        h = image.shape[0]
        integral = np.zeros((h+1,), dtype=dtype)
        integral[1:] = image.astype(dtype, copy=False).cumsum(0)
    else:
        raise ValueError("Поддерживаются только массивы с ndim от 1 до 3")

    return integral

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
    image = gauss_blur(image, blur_ksize)

    # Приведение к значениям яркости (градациям серого)
    if image.ndim != 2:
        image = np.dot(image, BGR_COEFFS)

    # Предвычисление интегрального изображения для яркости
    integral = integral_image(image)

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

def _get_area_integral_sum(integral: MatLike, coord: tuple[tuple[int, int], tuple[int, int]]) -> MatLike|num_dtype:
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

    x_points = np.floor(np.arange(0, x_size+1, x_step)).astype(int, copy=False)
    x_starts = x_points[:-1]
    x_ends = x_points[1:]
    y_points = np.floor(np.arange(0, y_size+1, y_step)).astype(int, copy=False)
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
