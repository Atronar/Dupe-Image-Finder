'''Функции, позволяющие сравнивать изображения на схожесть.
Алгоритм сравнения основан на анализе средней интенсивности изображения и его четвертей.

intensities() переводит изображение в вектор интенсивностей, который и используется для сравнения
similarity() указывает уровень близости между двумя векторами
similarity_images() указывает уровень близости между двумя изображениями

Также можно пользоваться более простыми готовыми функциями
is_similar_images() и is_similar()
которые вернут булево значение "похоже/не похоже" для двух изображений или векторов соответственно
'''
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, Generator, Iterable, Literal, TYPE_CHECKING, overload
from io import IOBase, TextIOBase
import numpy as np
import numpy.typing as npt


# Инициализация библиотек обработки изображений
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter, convolve as cp_convolve
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# MatLike нужен только для аннотаций. Без OpenCV подставляем Any,
# чтобы импорт библиотеки не зависел от этой необязательной зависимости.
if TYPE_CHECKING:
    from cv2.typing import MatLike
else:
    MatLike = Any
try:
    import pyvips
    VIPS_AVAILABLE = True
except ImportError:
    VIPS_AVAILABLE = False
try:
    import PIL.Image as pil
    import PIL.ImageOps as pil_ops
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

num_dtype = np.float32
num_integral_dtype = np.float32 | np.float64 | np.uint64 | np.int64
Vector = npt.NDArray[num_dtype]
npImageArray = npt.NDArray[num_dtype]
npIntegralImageArray = npt.NDArray[num_integral_dtype]
if TYPE_CHECKING:
    # Для проверки типов достаточно унаследовать от ndarray:
    # конкретный тип CuPy здесь не важен.
    class cpImageArray(np.ndarray): ...
    class cpIntegralImageArray(np.ndarray): ...
elif CUPY_AVAILABLE:
    cpImageArray = cp.ndarray
    cpIntegralImageArray = cp.ndarray
else:
    cpImageArray = np.ndarray
    cpIntegralImageArray = np.ndarray

# Коэффициенты яркости Y преобразования sRGB -> xyY для компонент BGR (ITU-R BT.709)
if CUPY_AVAILABLE:
    BGR_COEFFS_CUPY = cp.array([0.072186, 0.715158, 0.212656], dtype=num_dtype)
BGR_COEFFS = np.array([0.072186, 0.715158, 0.212656], dtype=num_dtype)

def to_gpu(arr: npImageArray) -> npImageArray | cpImageArray:
    if CUPY_AVAILABLE:
        mem = cp.cuda.Device().mem_info
        if arr.nbytes > mem[0] * 0.9:  # Не использовать более 90% свободной памяти
            return arr
        return cp.asarray(arr, dtype=num_dtype)
    return arr

def _is_cupy_array(arr: object) -> bool:
    """Проверяет, что массив является массивом CuPy.

    Проверка безопасна и при отсутствии CuPy: имя cp в этом случае
    не вычисляется благодаря короткому замыканию.
    """
    return CUPY_AVAILABLE and isinstance(arr, cp.ndarray)  # type: ignore[name-defined]

def from_gpu(arr: npImageArray | cpImageArray) -> npImageArray:
    return cp.asnumpy(arr) if _is_cupy_array(arr) else arr

def _is_binary_file_like(obj: object) -> bool:
    """Проверяет, что объект является бинарным файловым объектом.

    typing.BinaryIO нельзя использовать в isinstance: это только
    аннотация. IOBase покрывает BytesIO, BufferedIOBase и RawIOBase.
    """
    return isinstance(obj, IOBase) and not isinstance(obj, TextIOBase)

def _normalize_image_range(image: npt.NDArray[Any]) -> npImageArray:
    """Приводит изображение к float32 в диапазоне [0, 255].

    Разные декодеры возвращают разные типы и диапазоны: uint8 — 0…255,
    uint16 — 0…65535, float — 0…1 или 0…255. Для сравнения векторов
    важно, чтобы одна и та же картинка после любого декодера давала
    одинаковые значения.
    """
    if image.dtype == np.uint8:
        return image.astype(num_dtype, copy=False)
    if image.dtype == np.uint16:
        return image.astype(num_dtype) / 257.0
    if image.dtype == np.int16:
        return np.clip(image, 0, 65535).astype(num_dtype) / 257.0
    if np.issubdtype(image.dtype, np.floating):
        result = image.astype(num_dtype, copy=False)
        if result.size == 0:
            return result
        max_value = float(np.max(result))
        if max_value <= 1.0 + 1e-6:
            result = result * 255.0
        elif max_value > 255.0:
            result = result * (255.0 / max_value)
        return np.clip(result, 0.0, 255.0).astype(num_dtype, copy=False)
    # Целочисленные типы больше 16 бит и прочие: масштабируем по максимуму.
    if image.size == 0:
        return image.astype(num_dtype)
    max_value = float(np.max(image))
    if max_value > 255.0:
        return (image.astype(np.float64) * (255.0 / max_value)).astype(num_dtype)
    return image.astype(num_dtype)

def _read_image_header(source: str | Path | BinaryIO, size: int = 262144) -> bytes:
    """Читает начало файла, не сдвигая позицию файлового объекта."""
    if _is_binary_file_like(source):
        position = source.tell()
        try:
            return source.read(size)
        finally:
            source.seek(position)
    with open(source, "rb") as image_file:
        return image_file.read(size)

def _get_exif_orientation_tiff(tiff: bytes) -> int:
    """Возвращает Orientation (тег 0x0112) из TIFF/EXIF-блока."""
    if len(tiff) < 8:
        return 1
    if tiff[:2] == b"II":
        byte_order = "little"
    elif tiff[:2] == b"MM":
        byte_order = "big"
    else:
        return 1
    if int.from_bytes(tiff[2:4], byte_order) != 42:
        return 1
    ifd_offset = int.from_bytes(tiff[4:8], byte_order)
    if ifd_offset + 2 > len(tiff):
        return 1
    entries = int.from_bytes(tiff[ifd_offset:ifd_offset + 2], byte_order)
    position = ifd_offset + 2
    for _ in range(entries):
        if position + 12 > len(tiff):
            break
        tag = int.from_bytes(tiff[position:position + 2], byte_order)
        field_type = int.from_bytes(tiff[position + 2:position + 4], byte_order)
        count = int.from_bytes(tiff[position + 4:position + 8], byte_order)
        if tag == 0x0112 and count >= 1:
            if field_type == 3:  # SHORT
                return int.from_bytes(tiff[position + 8:position + 10], byte_order)
            if field_type == 4:  # LONG
                return int.from_bytes(tiff[position + 8:position + 12], byte_order)
        position += 12
    return 1

def _get_exif_orientation_jpeg(jpeg: bytes) -> int:
    """Ищет Orientation в APP1-сегменте JPEG."""
    position = 2  # пропускаем SOI
    while position + 4 <= len(jpeg):
        if jpeg[position] != 0xFF:
            position += 1
            continue
        marker = jpeg[position + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            position += 2
            continue
        length = int.from_bytes(jpeg[position + 2:position + 4], "big")
        if length < 2:
            return 1
        segment_end = position + 2 + length
        if marker == 0xE1 and jpeg[position + 4:position + 10] == b"Exif\x00\x00":
            return _get_exif_orientation_tiff(jpeg[position + 10:segment_end])
        if marker == 0xDA:  # начало сжатых данных
            return 1
        position = segment_end
    return 1

def _get_exif_orientation_png(png: bytes) -> int:
    """Ищет Orientation в PNG-чанке eXIf."""
    position = 8  # пропускаем сигнатуру PNG
    while position + 8 <= len(png):
        length = int.from_bytes(png[position:position + 4], "big")
        chunk_type = png[position + 4:position + 8]
        if chunk_type == b"eXIf":
            return _get_exif_orientation_tiff(png[position + 8:position + 8 + length])
        if chunk_type == b"IEND":
            break
        position += 12 + length
    return 1

def _get_exif_orientation(image_data: bytes) -> int:
    """Возвращает EXIF Orientation (1, если тега нет или формат неизвестен)."""
    try:
        if image_data[:2] == b"\xff\xd8":
            return _get_exif_orientation_jpeg(image_data)
        if image_data[:8] == b"\x89PNG\r\n\x1a\n":
            return _get_exif_orientation_png(image_data)
        if image_data[:4] in (b"II*\x00", b"MM\x00*"):
            return _get_exif_orientation_tiff(image_data)
    except (IndexError, ValueError):
        pass
    return 1

def _apply_exif_orientation(image: npt.NDArray[Any], orientation: int) -> npt.NDArray[Any]:
    """Поворачивает/отражает изображение согласно EXIF Orientation."""
    if orientation == 2:
        image = image[:, ::-1]
    elif orientation == 3:
        image = image[::-1, ::-1]
    elif orientation == 4:
        image = image[::-1, :]
    elif orientation == 5:
        image = np.swapaxes(image, 0, 1)
    elif orientation == 6:
        image = np.rot90(image, k=-1)
    elif orientation == 7:
        image = np.flip(np.swapaxes(image, 0, 1), axis=(0, 1))
    elif orientation == 8:
        image = np.rot90(image, k=1)
    else:
        return image
    return np.ascontiguousarray(image)

def open_image(image_path: str|Path|BinaryIO) -> npImageArray:
    """Функция, открывающая изображение по его пути, возвращается матрица пикселей в BGR
    image_path: путь к изображению
    """
    try:
        gif = False
        if CV2_AVAILABLE:
            return open_image_cv(image_path)
    except FileNotFoundError:
        gif = True
        pass
    if VIPS_AVAILABLE:
        return open_image_vips(image_path)
    if PIL_AVAILABLE:
        return open_image_pil(image_path)
    if gif:
        raise ModuleNotFoundError(
            "Для работы с данным типом изображений необходимо "
            "установить хотя бы одну из этих библиотек: "
            "pyvips, pil"
        )
    else:
        raise ModuleNotFoundError(
            "Необходимо установить хотя бы одну из этих библиотек: "
            "pyvips, pil, opencv"
        )

def open_image_vips(
    image_path: str | Path | BinaryIO,
    file_format: str | None = None
) -> npImageArray:
    """Открытие изображения в BGR с помощью libvips

    image_path: путь к изображению
    file_format: (необязательно) формат входного файла

    Возвращается ndarray[float32] — матрица пикселей в BGR
    """
    if not VIPS_AVAILABLE:
        raise ModuleNotFoundError(
            "pyvips не установлен. "
            "Необходимо использовать функцию open_image()"
        )

    if _is_binary_file_like(image_path):
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

    # libvips умеет применять EXIF-ориентацию самостоятельно.
    image = image.autorot()

    # удаляем альфа-канал, заменяя его белым цветом
    if image.hasalpha():
        # У 16-битных и float-изображений белый — не 255.
        if image.format == "ushort":
            background = 65535.0
        elif image.format in ("float", "double"):
            background = 1.0 if image.max() <= 1.0 else 255.0
        else:
            background = 255.0
        image = image.flatten(background=background)

    # Обработка Grayscale или одноканальных данных
    if image.bands == 1 or image.interpretation in ["b-w", "greyscale"]:
        array = image.numpy(dtype=num_dtype)
        if array.ndim == 3:
            array = array[..., 0]  # удаление оси каналов (H, W)
        return _normalize_image_range(array)
    # Конвертация CMYK → RGB
    elif image.interpretation == "cmyk":
        # Используем встроенный профиль sRGB
        image = image.icc_transform("srgb")
    # Обработка других не-RGB случаев (например, многоканальные не-RGB)
    elif "rgb" not in image.interpretation:
        # Конвертация в RGB через промежуточное пространство
        image = image.colourspace("srgb")
    # Конвертация RGB → BGR
    return _normalize_image_range(image[2::-1].numpy(dtype=num_dtype))

def open_image_cv(image_path: str|Path|BinaryIO) -> npImageArray:
    """Открытие изображения в BGR с помощью opencv

    image_path: путь к изображению

    Возвращается ndarray[float32] — матрица пикселей в BGR
    """
    if not CV2_AVAILABLE:
        raise ModuleNotFoundError(
            "opencv не установлен. "
            "Необходимо использовать функцию open_image()"
        )

    def load_image(image_path: str|Path|BinaryIO, flags: int) -> MatLike:
        if _is_binary_file_like(image_path):
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
    # IMREAD_UNCHANGED не применяет EXIF-ориентацию, поэтому делаем это сами.
    orientation = _get_exif_orientation(_read_image_header(image_path))
    img = _apply_exif_orientation(img, orientation)
    # Приводим к единому диапазону: uint16 и float не должны попадать
    # в расчёты как 0…65535 или 0…1.
    img = _normalize_image_range(img)

    channels = 1 if img.ndim == 2 else img.shape[2]
    is_grayscale = channels < 3
    has_alpha = channels in (2, 4)

    # Изображения без альфа-канала возвращаем как просто BGR/Grayscale
    if not has_alpha:
        if (
            not is_grayscale
            and np.array_equal(img[..., 0], img[..., 1])
            and np.array_equal(img[..., 0], img[..., 2])
        ):
            img = img[..., 0]
        return img

    # Приведение фактического серого из RGBA в градации серого без потери альфа-канала
    # Сравнение значений каналов
    if (
        not is_grayscale
        and np.array_equal(img[..., 0], img[..., 1])
        and np.array_equal(img[..., 0], img[..., 2])
    ):
        # Берём каналы без view: для uint16 view(np.uint8) даёт
        # неверную форму и портит значения.
        img = img[..., [0, 3]]
        is_grayscale = True

    # Удаление альфа-канала
    # Простая конвертация заменяет альфу на чёрный вместо белого, что недопустимо
    # Нормализация альфа-канала в [0, 1]
    alpha = img[..., -1][..., np.newaxis] / 255.0
    color = img[..., :-1]
    if is_grayscale:
        white = 255
    else:
        white = np.array([255.0, 255.0, 255.0], dtype=num_dtype)
    image = color * alpha + white * (1 - alpha)
    # squeeze() удалял бы и пространственные оси у изображений 1×N,
    # поэтому убираем только канал.
    if is_grayscale:
        return image[..., 0].astype(num_dtype, copy=False)
    return image.astype(num_dtype, copy=False)

def open_image_pil(image_path: str|Path|BinaryIO) -> npImageArray:
    """Открытие изображения в BGR с помощью PIL

    image_path: путь к изображению

    Возвращается ndarray[float32] — матрица пикселей в BGR
    """
    if not PIL_AVAILABLE:
        raise ModuleNotFoundError(
            "pillow не установлен. "
            "Необходимо использовать функцию open_image()"
        )

    with pil.open(image_path) as img:
        # Pillow по умолчанию не применяет EXIF-ориентацию.
        img = pil_ops.exif_transpose(img)
        convert_to = None
        mode = img.mode
        # I;16, I и F — одноканальные изображения с расширенным диапазоном.
        is_grayscale = mode in ['L', 'LA'] or mode.startswith('I') or mode == 'F'
        has_alpha = mode in ['RGBA', 'LA']
        is_rgb = 'RGB' in mode

        # Изображения с палитрой и альфа
        if mode in ('P', 'PA'):
            palette_mode = img.palette.mode if img.palette is not None else 'RGB'
            if "transparency" in img.info or palette_mode in ['RGBA', 'LA']:
                has_alpha = True
                if palette_mode == 'LA':
                    convert_to = 'LA'
                    is_grayscale = True
                    is_rgb = False
                else:
                    convert_to = 'RGBA'
                    is_grayscale = False
                    is_rgb = True
            elif palette_mode == 'L':
                convert_to = 'L'
                is_grayscale = True
                is_rgb = False
            elif palette_mode == 'RGB':
                convert_to = 'RGB'
                is_grayscale = False
                is_rgb = True
        elif 'transparency' in img.info and not has_alpha:
            # Например, RGB с прозрачным цветом: Pillow умеет это только через RGBA.
            convert_to = 'RGBA'
            has_alpha = True
            is_grayscale = False
            is_rgb = True

        # Сначала конвертируем палитру/специальные режимы, и только потом
        # анализируем каналы: у режима P массив содержит индексы палитры,
        # а не значения RGB.
        if convert_to:
            img = img.convert(convert_to)

        arr = _normalize_image_range(np.asarray(img))

        # Приведение фактического серого из RGB(A) в градации серого
        # без потери альфа-канала.
        if (
            is_rgb
            and arr.ndim == 3
            and arr.shape[2] >= 3
            and np.array_equal(arr[..., 0], arr[..., 1])
            and np.array_equal(arr[..., 0], arr[..., 2])
        ):
            convert_to = 'LA' if has_alpha else 'L'
            img = img.convert(convert_to)
            arr = _normalize_image_range(np.asarray(img))
            is_grayscale = True
            is_rgb = False

        # Удаление альфа-канала
        # Простая конвертация заменяет альфу на чёрный вместо белого, что недопустимо
        if has_alpha:
            alpha = arr[..., -1] / 255.0       # [0, 1]
            color = arr[..., :-1]              # [0, 255]
            if is_grayscale:
                image = color[..., 0] * alpha + 255.0 * (1 - alpha)
            else:
                white = np.array([255.0, 255.0, 255.0], dtype=num_dtype)
                image = color * alpha[..., np.newaxis] + white * (1 - alpha[..., np.newaxis])
        else:
            # Серый и RGB конвертировать не нужно
            if not (is_grayscale or is_rgb):
                img = img.convert('RGB')
                arr = _normalize_image_range(np.asarray(img))
                is_rgb = True
            image = arr

    if not is_grayscale:
        # RGB -> BGR
        image = image[..., ::-1]

    return np.ascontiguousarray(image, dtype=num_dtype)

@overload
def _gauss_blur(
    image: npImageArray,
    blur_ksize: int | tuple[int, int] = (3, 3),
    sigma: float = 0
) -> npImageArray: ...
@overload
def _gauss_blur(
    image: cpImageArray,
    blur_ksize: int | tuple[int, int] = (3, 3),
    sigma: float = 0
) -> cpImageArray: ...
def _gauss_blur(
    image: npImageArray | cpImageArray,
    blur_ksize: int | tuple[int, int] = (3, 3),
    sigma: float = 0,
) -> npImageArray | cpImageArray:
    """Размытие изображения по Гауссу

    image: массив изображения
    blur_ksize: размеры гауссова ядра, должно быть нечётным
    sigma: стандартное отклонение

    Возвращается массив размытого изображения
    """
    if CUPY_AVAILABLE and isinstance(image, cp.ndarray):
        return _gauss_blur_gpu(image, blur_ksize, sigma)
    if CV2_AVAILABLE:
        return _gauss_blur_cv(image, blur_ksize, sigma)
    if VIPS_AVAILABLE:
        return _gauss_blur_vips(image, blur_ksize, sigma)
    return _gauss_blur_custom(image, blur_ksize, sigma)

def _gauss_blur_vips(
    image: npImageArray,
    blur_ksize: int | tuple[int, int] = (3, 3),
    sigma: float = 0
) -> npImageArray:
    """Размытие изображения по Гауссу с помощью libvips

    image: массив изображения
    blur_ksize: размеры гауссова ядра, должно быть нечётным
    sigma: стандартное отклонение

    Возвращается массив размытого изображения
    """
    if not VIPS_AVAILABLE:
        raise ModuleNotFoundError(
            "pyvips не установлен. "
            "Необходимо использовать функцию gauss_blur()"
        )

    if isinstance(blur_ksize, tuple):
        blur_ksize = min(blur_ksize)
    if not blur_ksize:
        return image

    if np.isclose(sigma, 0):
        sigma = 0.15 * blur_ksize + 0.35
    min_ampl = np.exp(-(blur_ksize//2)**2 / (2 * sigma**2))

    blurred_image = pyvips.Image.new_from_array(image).gaussblur(sigma, min_ampl=min_ampl)
    return blurred_image.numpy(dtype=num_dtype)

def _gauss_blur_cv(
    image: npImageArray,
    blur_ksize: int | tuple[int, int] = (3, 3),
    sigma: float = 0
) -> npImageArray:
    """Размытие изображения по Гауссу с помощью opencv

    image: массив изображения
    blur_ksize: размеры гауссова ядра, должно быть нечётным
    sigma: стандартное отклонение

    Возвращается массив размытого изображения
    """
    if not CV2_AVAILABLE:
        raise ModuleNotFoundError(
            "opencv не установлен. "
            "Необходимо использовать функцию gauss_blur()"
        )

    if isinstance(blur_ksize, int):
        blur_ksize = (blur_ksize, blur_ksize)
    if not all(blur_ksize):
        return image
    return cv2.GaussianBlur(image, blur_ksize, sigma).astype(num_dtype, copy=False)

def _gauss_blur_gpu(
    image: cpImageArray,
    blur_ksize: int | tuple[int, int] = (3, 3),
    sigma: float = 0
) -> cpImageArray:
    if not CUPY_AVAILABLE:
        raise ModuleNotFoundError(
            "cupy не установлен. "
            "Необходимо использовать функцию gauss_blur()"
        )

    if isinstance(blur_ksize, tuple):
        blur_ksize = min(blur_ksize)
    if not blur_ksize:
        return image

    if np.isclose(sigma, 0):
        sigma = 0.15 * blur_ksize + 0.35

    return cp_gaussian_filter(image, sigma=sigma)

def _gauss_blur_custom(
    image: npImageArray,
    blur_ksize: int | tuple[int, int] = (3, 3),
    sigma: float = 0
) -> npImageArray:
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

    # kx — размер ядра по ширине (ось 1), ky — по высоте (ось 0).
    pad_x = kx // 2
    pad_y = ky // 2
    if image.ndim > 2:
        padded_image = np.pad(
            image,
            ((pad_y, pad_y), (pad_x, pad_x), (0, 0)),
            mode='reflect'
        )
    else:
        padded_image = np.pad(
            image,
            ((pad_y, pad_y), (pad_x, pad_x)),
            mode='reflect'
        )

    # Раздельная свёртка через скользящее окно: без Python-циклов
    # и без apply_along_axis, который был очень медленным.
    windows_x = np.lib.stride_tricks.sliding_window_view(
        padded_image, kx, axis=1
    )
    blurred_x = np.einsum('...i,i->...', windows_x, kernel_x, optimize=True)

    windows_y = np.lib.stride_tricks.sliding_window_view(
        blurred_x, ky, axis=0
    )
    blurred = np.einsum('...i,i->...', windows_y, kernel_y, optimize=True)

    return blurred.astype(num_dtype, copy=False)

@overload
def _integral_image(image: npImageArray, sq: Literal[False]) -> npIntegralImageArray: ...
@overload
def _integral_image(image: npImageArray, sq: Literal[True]) -> tuple[npIntegralImageArray, npIntegralImageArray]: ...
@overload
def _integral_image(image: cpImageArray, sq: Literal[False]) -> cpIntegralImageArray: ...
@overload
def _integral_image(image: cpImageArray, sq: Literal[True]) -> tuple[cpIntegralImageArray, cpIntegralImageArray]: ...
def _integral_image(
    image: npImageArray | cpImageArray,
    sq: bool = False
) -> (
    npIntegralImageArray
    | cpIntegralImageArray
    | tuple[npIntegralImageArray, npIntegralImageArray]
    | tuple[cpIntegralImageArray, cpIntegralImageArray]
):
    """
    Вычисляет интегральное изображение.

    image: массив изображения
    sq: флаг получения вместе с обычным интегральным изображением,
        квадратичного интегрального изображениия

    Возвращается массив интегрального изображения с нулевой рамкой —
    кумулятивная сумма по каждому измерению
    """
    if CUPY_AVAILABLE and isinstance(image, cp.ndarray):
        return _integral_image_gpu(image, sq=sq)
    if CV2_AVAILABLE:
        return _integral_image_cv(image, sq=sq)
    return _integral_image_custom(image, sq=sq)

@overload
def _integral_image_cv(
    image: npImageArray,
    sq: Literal[False]
) -> npt.NDArray: ...
@overload
def _integral_image_cv(
    image: npImageArray,
    sq: Literal[True]
) -> tuple[npt.NDArray, npt.NDArray]: ...
def _integral_image_cv(
    image: npImageArray,
    sq: bool = False
) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
    """
    Вычисляет интегральное изображение с помощью opencv.

    image: массив изображения
    sq: флаг получения вместе с обычным интегральным изображением,
        квадратичного интегрального изображениия

    Возвращается массив интегрального изображения с нулевой рамкой —
    кумулятивная сумма по каждому измерению
    """
    if not CV2_AVAILABLE:
        raise ModuleNotFoundError(
            "opencv не установлен. "
            "Необходимо использовать функцию integral_image()"
        )
    if sq:
        return cv2.integral2(image)
    return cv2.integral(image)

@overload
def _integral_image_gpu(
    image: cpImageArray,
    sq: Literal[False]
) -> cpIntegralImageArray: ...
@overload
def _integral_image_gpu(
    image: cpImageArray,
    sq: Literal[True]
) -> tuple[cpIntegralImageArray, cpIntegralImageArray]: ...
@overload
def _integral_image_gpu(
    image: cpImageArray,
    sq: bool = False
) -> cpIntegralImageArray | tuple[cpIntegralImageArray, cpIntegralImageArray]: ...
def _integral_image_gpu(
    image: cpImageArray,
    sq: bool = False
) -> cpIntegralImageArray | tuple[cpIntegralImageArray, cpIntegralImageArray]:
    """
    Вычисляет интегральное изображение с помощью cupy.

    image: массив изображения
    sq: флаг получения вместе с обычным интегральным изображением,
        квадратичного интегрального изображениия

    Возвращается массив интегрального изображения с нулевой рамкой —
    кумулятивная сумма по каждому измерению
    """
    if not CUPY_AVAILABLE:
        raise ModuleNotFoundError(
            "cupy не установлен. "
            "Необходимо использовать функцию integral_image()"
        )
    # Выбор типа данных.
    # Для вещественных изображений используем float64: суммы квадратов
    # на больших изображениях в float32 теряют точность, из-за чего
    # дисперсия может стать отрицательной.
    kind = image.dtype.kind
    dtype = (
        cp.float64 if kind == 'f'
        else cp.uint64 if kind == 'u'
        else cp.int64
    )

    def get_integral(img: cpImageArray) -> cpIntegralImageArray:
        # Создаем нулевую рамку
        if img.ndim == 3:
            h, w, c = img.shape
            integral = cp.zeros((h+1, w+1, c), dtype=dtype)
            integral[1:, 1:, :] = cp.cumsum(cp.cumsum(img, axis=0), axis=1)
        elif img.ndim == 2:
            h, w = img.shape
            integral = cp.zeros((h+1, w+1), dtype=dtype)
            integral[1:, 1:] = cp.cumsum(cp.cumsum(img, axis=0), axis=1)
        elif img.ndim == 1:
            h = img.shape[0]
            integral = cp.zeros((h+1,), dtype=dtype)
            integral[1:] = cp.cumsum(img, axis=0)
        else:
            raise ValueError("Поддерживаются только массивы с ndim от 1 до 3")
        return integral

    if sq:
        image_sq = image ** 2
        return get_integral(image), get_integral(image_sq)
    return get_integral(image)

@overload
def _integral_image_custom(image: npImageArray, sq: Literal[False]) -> npIntegralImageArray: ...
@overload
def _integral_image_custom(image: npImageArray, sq: Literal[True]) -> tuple[npIntegralImageArray, npIntegralImageArray]: ...
def _integral_image_custom(
    image: npImageArray,
    sq: bool = False
) -> npIntegralImageArray | tuple[npIntegralImageArray, npIntegralImageArray]:
    """
    Вычисляет интегральное изображение с помощью чистого numpy.

    image: массив изображения
    sq: флаг получения вместе с обычным интегральным изображением,
        квадратичного интегрального изображениия

    Возвращается массив интегрального изображения с нулевой рамкой —
    кумулятивная сумма по каждому измерению
    """
    # Выбор типа данных.
    # Для вещественных изображений используем float64: суммы квадратов
    # на больших изображениях в float32 теряют точность, из-за чего
    # дисперсия может стать отрицательной.
    kind = image.dtype.kind
    dtype = (
        np.float64 if kind == 'f'
        else np.uint64 if kind == 'u'
        else np.int64
    )

    def get_integral(img: npImageArray) -> npIntegralImageArray:
        # Создаем нулевую рамку
        if img.ndim == 3:
            h, w, c = img.shape
            integral = np.zeros((h+1, w+1, c), dtype=dtype)
            integral[1:, 1:, :] = img.astype(dtype, copy=False).cumsum(0).cumsum(1)
        elif img.ndim == 2:
            h, w = img.shape
            integral = np.zeros((h+1, w+1), dtype=dtype)
            integral[1:, 1:] = img.astype(dtype, copy=False).cumsum(0).cumsum(1)
        elif img.ndim == 1:
            h = img.shape[0]
            integral = np.zeros((h+1,), dtype=dtype)
            integral[1:] = img.astype(dtype, copy=False).cumsum(0)
        else:
            raise ValueError("Поддерживаются только массивы с ndim от 1 до 3")
        return integral

    if sq:
        image_sq = image ** 2
        return get_integral(image), get_integral(image_sq)
    return get_integral(image)

@overload
def _sobel(image: npImageArray, dy: bool=False) -> npImageArray: ...
@overload
def _sobel(image: cpImageArray, dy: bool=False) -> cpImageArray: ...
def _sobel(image: npImageArray|cpImageArray, dy: bool=False) -> npImageArray|cpImageArray:
    """
    Вычисляет градиент Собеля для изображения.

    image: массив изображения
    dy: сменить направление градиента на вертикальное (по умолчанию — горизонтальное),
        считается только первая производная

    Возвращается массив градиента изображения
    """
    if dy:
        _dx, _dy = 0, 1
    else:
        _dx, _dy = 1, 0
    if CUPY_AVAILABLE and isinstance(image, cp.ndarray):
        return _sobel_gpu(image, _dx, _dy)
    if CV2_AVAILABLE:
        return _sobel_cv(image, _dx, _dy)
    return _sobel_custom(image, _dx, _dy)

def _sobel_cv(image: npImageArray, dx: int, dy: int) -> npImageArray:
    """
    Вычисляет градиент Собеля для изображения с помощью opencv.

    image: массив изображения
    dx, dy: производная по направлению градиента

    Возвращается массив градиента изображения
    """
    if not CV2_AVAILABLE:
        raise ModuleNotFoundError(
            "opencv не установлен. "
            "Необходимо использовать функцию _sobel()"
        )
    return cv2.Sobel(image, cv2.CV_32F, dx, dy, ksize=3).astype(num_dtype, copy=False)

def _sobel_gpu(image: cpImageArray, dx: int, dy: int) -> cpImageArray:
    """
    Вычисляет градиент Собеля для изображения с помощью cupy.

    image: массив изображения
    dx, dy: направление градиента, считается только первая производная

    Возвращается массив градиента изображения
    """
    if not CUPY_AVAILABLE:
        raise ModuleNotFoundError(
            "cupy не установлен. "
            "Необходимо использовать функцию _sobel()"
        )
    # Ядра Собеля
    sobel_x_kernel = cp.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=cp.float32)
    sobel_y_kernel = cp.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ], dtype=cp.float32)

    # Применение свертки
    result = cp.zeros_like(image)
    if dx > 0:
        result += cp_convolve(image, sobel_x_kernel, mode='reflect')
    elif dy > 0:
        result += cp_convolve(image, sobel_y_kernel, mode='reflect')
    return result

def _sobel_custom(image: npImageArray, dx: int, dy: int) -> npImageArray:
    """
    Вычисляет градиент Собеля для изображения с помощью чистого numpy.

    image: массив изображения
    dx, dy: направление градиента, считается только первая производная

    Возвращается массив градиента изображения
    """
    def _conv_1d_strided(
        image: npImageArray,
        kernel: npImageArray,
        axis: Literal[0, 1]
    ) -> npImageArray:
        shape = list(image.shape) + [3,]
        shape[axis] -= 2
        shape = tuple(shape)

        # Вычисление strides вручную
        strides = (image.strides[0], image.strides[1], image.strides[axis])

        # Создание "окон" через as_strided
        windows = np.lib.stride_tricks.as_strided(
            image,
            shape = shape,
            strides = strides
        )

        # Свертка и суммирование
        return np.sum(windows * kernel, axis=-1).astype(np.float32)

    pad_image = np.pad(image, pad_width=1, mode='reflect')

    # Ядра Собеля (разделенные)
    kernel_vert = np.array([1, 2, 1], dtype=np.float32)
    kernel_horiz = np.array([-1, 0, 1], dtype=np.float32)

    result = np.zeros_like(image, dtype=num_dtype)
    # Вертикальная свертка (ось Y)
    if dx > 0:
        temp = _conv_1d_strided(pad_image, kernel_vert, axis=0)
        result = _conv_1d_strided(temp, kernel_horiz, axis=1)

    # Горизонтальная свертка (ось X)
    elif dy > 0:
        temp = _conv_1d_strided(pad_image, kernel_horiz, axis=0)
        result = _conv_1d_strided(temp, kernel_vert, axis=1)

    return result

@overload
def _grad_integral_image(
    image: npImageArray
) -> npIntegralImageArray: ...
@overload
def _grad_integral_image(
    image: cpImageArray
) -> cpIntegralImageArray: ...
def _grad_integral_image(
    image: npImageArray | cpImageArray
) -> npIntegralImageArray | cpIntegralImageArray:
    """
    Вычисляет интегральное изображение градиента изображения.

    image: массив изображения

    Возвращается массив интегрального градиента изображения с нулевой рамкой —
    кумулятивная сумма по каждому измерению
    """
    grad_x = _sobel(image, dy=False)
    grad_y = _sobel(image, dy=True)
    if CUPY_AVAILABLE and isinstance(image, cp.ndarray):
        grad_mag = cp.sqrt(grad_x**2 + grad_y**2)
    else:
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return _integral_image(grad_mag, sq=False)

def intensities(
    image_path: str | Path | BinaryIO,
    partition_level: int = 2,
    blur_ksize: int | tuple[int, int] = (3, 3),
    use_threads: bool = True,
    use_gpu: bool = True
) -> list[Vector]:
    '''
    https://gist.github.com/liamwhite/b023cdba4738e911293a8c610b98f987
    Алгоритм основан на анализе изображения и его четвертей

    image_path: путь к изображению

    partition_level устанавливает размерность вектора, которая будет равна
    sum{n=1,partition_level}(n^2)
    В принципе, можно получать и одномерные вектора, но меньшая размерность приводит к тому,
    что вектора чаще будут считаться похожими

    blur_ksize: Размер ядра размытия Гаусса, (0,0) для отключения

    Возвращается список векторов, соответствующих изображению, полученному на входе
    Первый вектор основан на средней интенсивности
    Второй - на контрасте
    Третий - на текстуре
    '''
    image = open_image(image_path)

    if (
        # Для цветных менее (условно) 512x512 и серых менее (условно) 1536x1536
        # использование GPU может быть медленнее, чем CPU
        image.ndim < 3 and image.shape[0]*image.shape[1]<2359296
        or image.shape[0]*image.shape[1]<262144
    ):
        use_gpu = False

    if use_gpu:
        image_gpu = to_gpu(image)
        if not _is_cupy_array(image_gpu):
            use_gpu = False
    else:
        image_gpu = image

    # Размытие изображения
    image_gpu = _gauss_blur(image_gpu, blur_ksize)

    # Приведение к значениям яркости (градациям серого)
    if image.ndim != 2:
        if CUPY_AVAILABLE and use_gpu:
            image_gpu = cp.dot(image_gpu, BGR_COEFFS_CUPY)
        else:
            image_gpu = np.dot(image_gpu, BGR_COEFFS)

    # Предвычисление интегрального изображения для яркости и контраста
    integral, integral_sq = _integral_image(image_gpu, sq=True)

    # Вычисление интегрального градиента
    grad_integral = _grad_integral_image(image_gpu)

    # Высота и ширина всего изображения
    y_size, x_size = image.shape[:2]

    # Расчёт итогового вектора, характеризующего изображение
    features_list: list[Vector] = []
    features_integrals_list: list[tuple[npIntegralImageArray|cpIntegralImageArray, ...]] = [
        (integral,),
        (integral, integral_sq),
        (grad_integral,),
    ]
    if use_threads:
        with ThreadPoolExecutor() as executor:
            futures = []
            for current_integral in features_integrals_list:
                args = (current_integral, partition_level, (x_size, y_size), use_threads)
                futures.append(executor.submit(_process_calculate_features, args))
            for future in futures:
                features_list.append(future.result())
    else:
        for current_integral in features_integrals_list:
            if len(current_integral) == 2:
                features_list.append(_calculate_features_sq(
                    current_integral[0],
                    current_integral[1],
                    partition_level,
                    (x_size, y_size),
                    use_threads = use_threads
                ))
            else:
                features_list.append(_calculate_features(
                    current_integral[0],
                    partition_level,
                    (x_size, y_size),
                    use_threads = use_threads
                ))
    return features_list

def _process_calculate_features(
    args: tuple[
        tuple[npIntegralImageArray | cpIntegralImageArray, ...],
        int,
        tuple[int, int],
        bool
    ]
) -> Vector:
    '''Получение компонент вектора для текущего уровня разбиения через интегральное изображение
    Обёртка для _level_intensities()
    '''
    if len(args[0]) == 2:
        return _calculate_features_sq(*(args[0]), *(args[1:]))
    else:
        return _calculate_features(args[0][0], *(args[1:]))

def _calculate_features(
    integral: npIntegralImageArray | cpIntegralImageArray,
    partition_level: int,
    image_size: tuple[int, int],
    use_threads: bool = True
) -> Vector:
    '''Вычисление вектора, характеризующего изображение
    '''
    if CUPY_AVAILABLE and isinstance(integral, cp.ndarray):
        p = cp
    else:
        p = np
    # Не разбиваем на потоки единственную операцию
    if partition_level == 1:
        features = _level_intensities(
            integral,
            partition_level,
            image_size,
            image_size,
            use_threads = False
        )
        return from_gpu(features)

    # Высота и ширина всего изображения
    x_size, y_size = image_size

    # Выделяем память для вектора, характеризующего изображение
    features = p.empty(sum(level**2 for level in range(1, partition_level+1)), dtype=num_dtype)
    features_idx = 0

    if use_threads:
        with ThreadPoolExecutor() as executor:
            futures = []
            for current_partition_level in range(1, partition_level + 1):
                # Высота и ширина текущего разбиения
                part_size = (x_size / current_partition_level, y_size / current_partition_level)
                # Подготовка параметров уровня
                args = (integral, current_partition_level, image_size, part_size, use_threads)
                futures.append(executor.submit(_process_level_intensities, args))
            for future in futures:
                part_features = future.result()
                features[features_idx:features_idx+part_features.size] = part_features
                features_idx += part_features.size
    else:
        for current_partition_level in range(1, partition_level + 1):
            # Высота и ширина текущего разбиения
            part_size = (x_size / current_partition_level, y_size / current_partition_level)

            # Расчёт компонент итогового вектора
            part_features = _level_intensities(
                integral,
                current_partition_level,
                image_size,
                part_size,
                use_threads
            )
            features[features_idx:features_idx+part_features.size] = part_features
            features_idx += part_features.size

    return from_gpu(features)

@overload
def _process_level_intensities(
    args: tuple[
        npIntegralImageArray,
        int,
        tuple[int, int],
        tuple[float, float] | None,
        bool
    ]
) -> npImageArray: ...
@overload
def _process_level_intensities(
    args: tuple[
        cpIntegralImageArray,
        int,
        tuple[int, int],
        tuple[float, float] | None,
        bool
    ]
) -> cpImageArray: ...
def _process_level_intensities(
    args: tuple[
        npIntegralImageArray | cpIntegralImageArray,
        int,
        tuple[int, int],
        tuple[float, float] | None,
        bool
    ]
) -> npImageArray | cpImageArray:
    '''Получение компонент вектора для текущего уровня разбиения через интегральное изображение
    Обёртка для _level_intensities()
    '''
    return _level_intensities(*args)

@overload
def _level_intensities(
    integral: npIntegralImageArray,
    current_partition_level: int,
    image_size: tuple[int, int],
    part_size: tuple[float, float] | None = None,
    use_threads: bool = True
) -> npImageArray: ...
@overload
def _level_intensities(
    integral: cpIntegralImageArray,
    current_partition_level: int,
    image_size: tuple[int, int],
    part_size: tuple[float, float] | None = None,
    use_threads: bool = True
) -> cpImageArray: ...
def _level_intensities(
    integral: npIntegralImageArray | cpIntegralImageArray,
    current_partition_level: int,
    image_size: tuple[int, int],
    part_size: tuple[float, float] | None = None,
    use_threads: bool = True
) -> npImageArray | cpImageArray:
    '''Получение компонент вектора для текущего уровня разбиения через интегральное изображение
    '''
    if CUPY_AVAILABLE and isinstance(integral, cp.ndarray):
        p = cp
    else:
        p = np
    # Высота и ширина всего изображения
    x_size, y_size = image_size

    # Не разбиваем на потоки единственную операцию
    if current_partition_level == 1:
        feature = p.divide(integral[-1, -1], x_size*y_size)
        return p.asarray(feature, dtype=num_dtype).reshape(1)

    # Высота и ширина текущего разбиения
    if part_size is None:
        x_step = y_step = None
    else:
        x_step, y_step = part_size

    # Выделяем память для компонент вектора,
    # характеризующего изображение на текущем уровне разбиения
    features = p.empty(current_partition_level**2, dtype=num_dtype)
    features_idx = 0

    if use_threads:
        with ThreadPoolExecutor() as executor:
            futures = []

            for (y0, y1), (x0, x1) in generate_grid_coords(
                x_size,
                y_size,
                current_partition_level,
                x_step = x_step,
                y_step = y_step
            ):
                args = (integral, ((x0, x1), (y0, y1)))
                futures.append(executor.submit(_process_level_part_intensities, args))

            for future in futures:
                single_feature = future.result()
                features[features_idx] = single_feature
                features_idx += 1
    else:
        for (y0, y1), (x0, x1) in generate_grid_coords(
            x_size,
            y_size,
            current_partition_level,
            x_step = x_step,
            y_step = y_step
        ):
            # Расчёт компоненты итогового вектора
            features[features_idx] = _level_part_intensities(integral, ((x0, x1), (y0, y1)))
            features_idx += 1

    return features

def _process_level_part_intensities(
    args: tuple[
        npIntegralImageArray | cpIntegralImageArray,
        tuple[tuple[int, int], tuple[int, int]]
    ]
) -> float:
    '''Получение компоненты вектора для области через интегральное изображение
    Обёртка для _level_part_intensities()
    '''
    return _level_part_intensities(*args)

def _level_part_intensities(
    integral: npIntegralImageArray | cpIntegralImageArray,
    coordinates: tuple[tuple[int, int], tuple[int, int]]
) -> float:
    '''Получение компоненты вектора для области через интегральное изображение
    '''
    (x0, x1), (y0, y1) = coordinates
    # Площадь части
    area = (y1 - y0) * (x1 - x0)

    # Расчёт компоненты итогового вектора
    if area == 0:
        return 0.0
    # Быстрый расчёт суммы через интегральное изображение
    # и получение средней интенсивности в области
    sum_val = _get_area_integral_sum(integral, ((x0, x1), (y0, y1)))
    if CUPY_AVAILABLE and isinstance(integral, cp.ndarray):
        feature = cp.divide(sum_val, area)
    else:
        feature = np.divide(sum_val, area)
    return float(feature)

def _calculate_features_sq(
    integral: npIntegralImageArray | cpIntegralImageArray,
    integral_sq: npIntegralImageArray | cpIntegralImageArray,
    partition_level: int,
    image_size: tuple[int, int],
    use_threads: bool = True
) -> Vector:
    '''Вычисление вектора, характеризующего изображение
    '''
    if CUPY_AVAILABLE and isinstance(integral, cp.ndarray):
        p = cp
    else:
        p = np
    # Не разбиваем на потоки единственную операцию
    if partition_level == 1:
        features = _level_intensities_sq(
            integral,
            integral_sq,
            partition_level,
            image_size,
            image_size,
            use_threads = False
        )
        return from_gpu(features)

    # Высота и ширина всего изображения
    x_size, y_size = image_size

    # Выделяем память для вектора, характеризующего изображение
    features = p.empty(sum(level**2 for level in range(1, partition_level+1)), dtype=num_dtype)
    features_idx = 0

    if use_threads:
        with ThreadPoolExecutor() as executor:
            futures = []
            for current_partition_level in range(1, partition_level + 1):
                # Высота и ширина текущего разбиения
                part_size = (x_size / current_partition_level, y_size / current_partition_level)
                # Подготовка параметров уровня
                args = (integral, integral_sq, current_partition_level, image_size, part_size, use_threads)
                futures.append(executor.submit(_process_level_intensities_sq, args))
            for future in futures:
                part_features = future.result()
                features[features_idx:features_idx+part_features.size] = part_features
                features_idx += part_features.size
    else:
        for current_partition_level in range(1, partition_level + 1):
            # Высота и ширина текущего разбиения
            part_size = (x_size / current_partition_level, y_size / current_partition_level)

            # Расчёт компонент итогового вектора
            part_features = _level_intensities_sq(
                integral,
                integral_sq,
                current_partition_level,
                image_size,
                part_size,
                use_threads
            )
            features[features_idx:features_idx+part_features.size] = part_features
            features_idx += part_features.size

    return from_gpu(features)

@overload
def _process_level_intensities_sq(
    args: tuple[
        npIntegralImageArray,
        npIntegralImageArray,
        int,
        tuple[int, int],
        tuple[float, float] | None,
        bool
    ]
) -> npImageArray: ...
@overload
def _process_level_intensities_sq(
    args: tuple[
        cpIntegralImageArray,
        cpIntegralImageArray,
        int,
        tuple[int, int],
        tuple[float, float] | None,
        bool
    ]
) -> cpImageArray: ...
def _process_level_intensities_sq(
    args: tuple[
        npIntegralImageArray | cpIntegralImageArray,
        npIntegralImageArray | cpIntegralImageArray,
        int,
        tuple[int, int],
        tuple[float, float] | None,
        bool
    ]
) -> npImageArray | cpImageArray:
    '''Получение компонент вектора для текущего уровня разбиения через интегральное изображение
    Обёртка для _level_intensities_sq()
    '''
    return _level_intensities_sq(*args)

@overload
def _level_intensities_sq(
    integral: npIntegralImageArray,
    integral_sq: npIntegralImageArray,
    current_partition_level: int,
    image_size: tuple[int, int],
    part_size: tuple[float, float] | None = None,
    use_threads: bool = True
) -> npImageArray: ...
@overload
def _level_intensities_sq(
    integral: cpIntegralImageArray,
    integral_sq: cpIntegralImageArray,
    current_partition_level: int,
    image_size: tuple[int, int],
    part_size: tuple[float, float] | None = None,
    use_threads: bool = True
) -> cpImageArray: ...
def _level_intensities_sq(
    integral: npIntegralImageArray | cpIntegralImageArray,
    integral_sq: npIntegralImageArray | cpIntegralImageArray,
    current_partition_level: int,
    image_size: tuple[int, int],
    part_size: tuple[float, float] | None = None,
    use_threads: bool = True
) -> npImageArray | cpImageArray:
    '''Получение компонент вектора для текущего уровня разбиения через интегральное изображение
    '''
    if CUPY_AVAILABLE and isinstance(integral, cp.ndarray):
        p = cp
    else:
        p = np
    # Высота и ширина всего изображения
    x_size, y_size = image_size

    # Не разбиваем на потоки единственную операцию
    if current_partition_level == 1:
        area = x_size*y_size
        mean = integral[-1, -1] / area
        variance = integral_sq[-1, -1] / area - mean**2
        # Из-за погрешности вычислений дисперсия может оказаться
        # чуть меньше нуля; sqrt от неё дал бы NaN.
        feature = p.sqrt(p.maximum(variance, 0))
        return p.asarray(feature, dtype=num_dtype).reshape(1)

    # Высота и ширина текущего разбиения
    if part_size is None:
        x_step = y_step = None
    else:
        x_step, y_step = part_size

    # Выделяем память для компонент вектора,
    # характеризующего изображение на текущем уровне разбиения
    features = p.empty(current_partition_level**2, dtype=num_dtype)
    features_idx = 0

    if use_threads:
        with ThreadPoolExecutor() as executor:
            futures = []

            for (y0, y1), (x0, x1) in generate_grid_coords(
                x_size,
                y_size,
                current_partition_level,
                x_step = x_step,
                y_step = y_step
            ):
                args = (integral, integral_sq, ((x0, x1), (y0, y1)))
                futures.append(executor.submit(_process_level_part_intensities_sq, args))

            for future in futures:
                single_feature = future.result()
                features[features_idx] = single_feature
                features_idx += 1
    else:
        for (y0, y1), (x0, x1) in generate_grid_coords(
            x_size,
            y_size,
            current_partition_level,
            x_step = x_step,
            y_step = y_step
        ):
            # Расчёт компоненты итогового вектора
            features[features_idx] = _level_part_intensities_sq(integral, integral_sq, ((x0, x1), (y0, y1)))
            features_idx += 1

    return features

def _process_level_part_intensities_sq(
    args: tuple[
        npIntegralImageArray | cpIntegralImageArray,
        npIntegralImageArray | cpIntegralImageArray,
        tuple[tuple[int, int], tuple[int, int]]
    ]
) -> float:
    '''Получение компоненты вектора для области через интегральное изображение
    Обёртка для _level_part_intensities_sq()
    '''
    return _level_part_intensities_sq(*args)

def _level_part_intensities_sq(
    integral: npIntegralImageArray | cpIntegralImageArray,
    integral_sq: npIntegralImageArray | cpIntegralImageArray,
    coordinates: tuple[tuple[int, int], tuple[int, int]]
) -> float:
    '''Получение компоненты вектора для области через интегральное изображение
    '''
    (x0, x1), (y0, y1) = coordinates
    # Площадь части
    area = (y1 - y0) * (x1 - x0)

    # Расчёт компоненты итогового вектора
    if area == 0:
        return 0.0
    # Быстрый расчёт суммы через интегральное изображение
    # и получение средней интенсивности в области
    sum_val = _get_area_integral_sum(integral, coordinates)
    sum_sq = _get_area_integral_sum(integral_sq, coordinates)
    if CUPY_AVAILABLE and isinstance(integral, cp.ndarray):
        mean = sum_val / area
        variance = sum_sq / area - mean**2
        feature = cp.sqrt(cp.maximum(variance, 0))
    else:
        mean = sum_val / area
        variance = sum_sq / area - mean**2
        feature = np.sqrt(np.maximum(variance, 0))
    return float(feature)

def _get_area_integral_sum(
    integral: npIntegralImageArray | cpIntegralImageArray,
    coord: tuple[tuple[int, int], tuple[int, int]]
) -> num_integral_dtype:
    '''Получение суммы пикселей области изображения через интегральную сумму
    '''
    (x0, x1), (y0, y1) = coord
    return integral[y1, x1] - integral[y1, x0] - integral[y0, x1] + integral[y0, x0]

def generate_grid_coords(
    x_size: int,
    y_size: int,
    side_partitions_num: int = 2,
    *,
    x_step: float | None = None,
    y_step: float | None = None
) -> Generator[tuple[tuple[int, int], tuple[int, int]], None, None]:
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

def division_on_partitions_iter(image: npImageArray, side_partitions_num: int = 2):
    '''Разбиение изображения равномерной сеткой на side_partitions_num^2 частей
    '''
    # Чистые размеры части без округлений
    partition_x_size_raw = image.shape[1] / side_partitions_num
    partition_y_size_raw = image.shape[0] / side_partitions_num

    for y, x in generate_grid_coords(
        image.shape[1],
        image.shape[0],
        side_partitions_num,
        x_step = partition_x_size_raw,
        y_step = partition_y_size_raw
    ):
        yield image[y[0]:y[1], x[0]:x[1]]

def get_rectangles_iter(image: npImageArray, partition_level: int = 2):
    '''Последовательное всё более дробное разбиение изображения на части и получение этих частей.
    Разбиение идёт от возврата самого изображения до сетки со стороной partition_level частей
    '''
    for current_partition_level in range(1, partition_level + 1):
        yield from division_on_partitions_iter(
            image,
            side_partitions_num = current_partition_level
        )

def rect_sum(rect: npImageArray) -> float:
    '''Средняя относительная яркость области.

    Для цветного изображения каналы BGR взвешиваются коэффициентами
    BGR_COEFFS; для изображения с альфа-каналом альфа накладывается на белый
    фон. Для серого изображения возвращается обычное среднее.
    '''
    if rect.size == 0:
        return 0.0

    if rect.ndim == 2:
        return float(np.mean(rect, dtype=np.float64))

    channels = rect.shape[2]
    if channels == 1:
        return float(np.mean(rect[..., 0], dtype=np.float64))

    if channels < 3:
        return float(np.mean(rect, dtype=np.float64))

    # Для расчёта яркости используем первые три канала BGR.
    color = rect[..., :3].astype(np.float64, copy=False)
    if channels >= 4:
        alpha = rect[..., 3:4].astype(np.float64, copy=False) / 255.0
        color = color * alpha + 255.0 * (1.0 - alpha)

    mean_bgr = color.mean(axis=(0, 1))
    return float(np.dot(mean_bgr, BGR_COEFFS))

def intensities_iter(
    image_list: Iterable[str | Path | BinaryIO],
    **kwargs
) -> Generator[list[Vector], None, None]:
    '''Итератор, возвращающий вектора для набора полученных изображений
    '''
    if kwargs.get("use_threads", True):
        # partial сохраняет имена параметров, а executor.map — порядок изображений.
        process = partial(intensities, **kwargs)
        with ThreadPoolExecutor() as executor:
            yield from executor.map(process, image_list)
    else:
        for image in image_list:
            yield intensities(image, **kwargs)

def similarity(vector1: Vector|tuple|list, vector2: Vector|tuple|list) -> float:
    '''Уровень похожести между векторами.
    0 - идентичные, 1 - максимально разные

    Векторы должны быть одномерными, непустыми и конечными.
    '''
    # Преобразование необходимо для вычисления разницы векторов
    vector1 = np.asarray(vector1, dtype=num_dtype)
    vector2 = np.asarray(vector2, dtype=num_dtype)

    if vector1.ndim != 1 or vector2.ndim != 1:
        raise ValueError("Векторы должны быть одномерными")
    if vector1.shape != vector2.shape:
        raise ValueError(
            f"Размеры векторов не совпадают: {vector1.shape} != {vector2.shape}"
        )
    if vector1.size == 0:
        raise ValueError("Векторы не должны быть пустыми")
    if not np.isfinite(vector1).all() or not np.isfinite(vector2).all():
        raise ValueError("Векторы содержат NaN или бесконечные значения")

    if np.array_equal(vector1, vector2):
        return 0.0

    norm_sum = np.sqrt(np.dot(vector1, vector1)) + np.sqrt(np.dot(vector2, vector2))

    if np.isclose(norm_sum, 0):
        # Неопределённость 0/0, но так как distance=0, то идентичны
        return 0.0

    diff = vector1 - vector2
    distance = np.sqrt(np.dot(diff, diff))

    return float(distance/norm_sum)

def is_similar(vector1: Vector, vector2: Vector, threshold: float=0.1) -> bool:
    '''Похожи ли два вектора
    threshold: уровень разницы, больше которого вектора считаются разными
    Возвращается bool
    '''
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Порог должен быть конечным неотрицательным числом")
    return similarity(vector1,vector2) < threshold

def similarity_images(
    image1: str | Path | BinaryIO,
    image2: str | Path | BinaryIO,
    **kwargs
) -> list[float]:
    '''Уровень похожести между изображениями.
    0 - идентичные, 1 - максимально разные
    '''
    if kwargs.get("use_threads", True):
        with ThreadPoolExecutor(2) as executor:
            future1 = executor.submit(intensities, image1, **kwargs)
            future2 = executor.submit(intensities, image2, **kwargs)
            vectors1 = future1.result()
            vectors2 = future2.result()
    else:
        vectors1 = intensities(image1, **kwargs)
        vectors2 = intensities(image2, **kwargs)
    return [similarity(vectors1[vector_num], vectors2[vector_num]) for vector_num in range(len(vectors1))]

def is_similar_images(
    image1: str | Path | BinaryIO,
    image2: str | Path | BinaryIO,
    threshold: float = 0.1,
    **kwargs
) -> bool:
    '''Похожи ли два изображения
    threshold: уровень разницы, больше которого изображения считаются разными
    Возвращается bool
    '''
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Порог должен быть конечным неотрицательным числом")
    return all(similarity_level < threshold for similarity_level in similarity_images(image1, image2, **kwargs))
