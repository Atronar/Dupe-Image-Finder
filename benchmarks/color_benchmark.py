"""Сравнение бэкендов цветового конвейера.

Скрипт запускается вручную:

    python benchmarks/color_benchmark.py

Для каждого размера изображения печатается медианное время отдельных
операций: декодирования, размытия, преобразования BGR → Lab, цветовой
доли, интегрального изображения и сборки цветового вектора. В конце
сравнивается цветовой конвейер целиком для профилей cv, vips, Pillow
и CuPy. Абсолютные значения зависят от машины.
"""
from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duplicatefinder as df


def measure(
    function: Callable[[], object],
    repeat: int = 7,
    number: int = 1,
    gpu: bool = False,
) -> float:
    """Возвращает медианное время выполнения в секундах."""
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        for _ in range(number):
            function()
        if gpu:
            df.cp.cuda.Stream.null.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def format_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.1f} мкс"
    return f"{seconds * 1e3:8.2f} мс"


def make_image(path: Path, size: int) -> None:
    """Создаёт цветное изображение с шумом."""
    cv2 = __import__("cv2")
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("не удалось записать тестовое изображение")


def ab_from_pillow(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Экспериментальный Pillow-бэкенд Lab.

    В библиотеке такого бэкенда нет: Pillow умеет конвертировать RGB → LAB
    только при наличии ImageCms. Функция нужна, чтобы измерить скорость
    и точность такого варианта, если он доступен.
    """
    pil = __import__("PIL.Image", fromlist=["Image"])
    rgb = np.clip(image[..., ::-1], 0, 255).astype(np.uint8)
    lab = pil.fromarray(rgb, "RGB").convert("LAB")
    array = np.asarray(lab, dtype=np.float32)
    return array[..., 1] - 128.0, array[..., 2] - 128.0


PILLOW_LAB_ERROR: str | None = None
try:
    ab_from_pillow(np.zeros((4, 4, 3), dtype=np.float32))
except Exception as error:  # ImageCms может отсутствовать в сборке Pillow
    PILLOW_LAB_ERROR = f"{type(error).__name__}: {error}"


def color_pipeline(
    profile: str,
    image_path: Path,
    image_size: tuple[int, int],
    level: int = 2,
) -> np.ndarray:
    """Цветовой конвейер целиком для выбранного CPU-профиля."""
    if profile == "cv":
        image = df.open_image_cv(str(image_path))
        image = df._gauss_blur_cv(image, (3, 3))
        lab_a, lab_b = df._get_ab_from_BGR_cv(image)
    elif profile == "vips":
        image = df.open_image_vips(str(image_path))
        image = df._gauss_blur_vips(image, (3, 3))
        lab_a, lab_b = df._get_ab_from_BGR_vips(image)
    else:
        # У Pillow нет ни размытия, ни Lab, поэтому в этом профиле он
        # отвечает только за декодирование, остальное делает NumPy.
        image = df.open_image_pil(str(image_path))
        image = df._gauss_blur_custom(image, (3, 3))
        lab_a, lab_b = df._get_ab_from_BGR_custom(image)
    df._color_presence(lab_a, lab_b)
    return df._calculate_color_vector(
        lab_a, lab_b, level, image_size
    )


def color_pipeline_gpu(
    image: np.ndarray,
    image_size: tuple[int, int],
    level: int = 2,
) -> np.ndarray:
    """Цветовой конвейер целиком на GPU (без декодирования)."""
    image_gpu = df.to_gpu(image)
    image_gpu = df._gauss_blur_gpu(image_gpu, (3, 3))
    lab_a, lab_b = df._get_ab_from_BGR_gpu(image_gpu)
    df._color_presence(lab_a, lab_b)
    return df._calculate_color_vector(
        lab_a, lab_b, level, image_size
    )


def print_header() -> None:
    print(
        f"NumPy: {np.__version__}, CuPy: {df.CUPY_AVAILABLE}, "
        f"OpenCV: {df.CV2_AVAILABLE}, pyvips: {df.VIPS_AVAILABLE}, "
        f"Pillow: {df.PIL_AVAILABLE}"
    )
    if PILLOW_LAB_ERROR is not None:
        print(f"Pillow Lab недоступен: {PILLOW_LAB_ERROR}")


def benchmark_size(size: int, temp_dir: Path) -> None:
    path = temp_dir / f"color-{size}.png"
    make_image(path, size)
    image = df.open_image_cv(str(path))
    print(f"\n=== {size}x{size}, float32 BGR ===")

    print("\n-- декодирование --")
    if df.CV2_AVAILABLE:
        print("open_image_cv  ", format_time(
            measure(lambda: df.open_image_cv(str(path)))
        ))
    if df.VIPS_AVAILABLE:
        print("open_image_vips", format_time(
            measure(lambda: df.open_image_vips(str(path)))
        ))
    if df.PIL_AVAILABLE:
        print("open_image_pil ", format_time(
            measure(lambda: df.open_image_pil(str(path)))
        ))

    print("\n-- размытие 3x3 --")
    if df.CV2_AVAILABLE:
        print("gauss_cv       ", format_time(
            measure(lambda: df._gauss_blur_cv(image, (3, 3)))
        ))
    if df.VIPS_AVAILABLE:
        print("gauss_vips     ", format_time(
            measure(lambda: df._gauss_blur_vips(image, (3, 3)))
        ))
    print("gauss_custom   ", format_time(
        measure(lambda: df._gauss_blur_custom(image, (3, 3)), repeat=3)
    ))

    print("\n-- Lab a/b --")
    if df.CV2_AVAILABLE:
        reference_a, reference_b = df._get_ab_from_BGR_cv(image)
    else:
        reference_a, reference_b = df._get_ab_from_BGR_custom(image)
    backends: list[tuple[str, Callable[[], tuple[np.ndarray, np.ndarray]]]] = []
    if df.CV2_AVAILABLE:
        backends.append((
            "lab_cv        ",
            lambda: df._get_ab_from_BGR_cv(image),
        ))
    if df.VIPS_AVAILABLE:
        backends.append((
            "lab_vips      ",
            lambda: df._get_ab_from_BGR_vips(image),
        ))
    backends.append((
        "lab_custom    ",
        lambda: df._get_ab_from_BGR_custom(image),
    ))
    if PILLOW_LAB_ERROR is None:
        backends.append(("lab_pillow    ", lambda: ab_from_pillow(image)))
    for name, function in backends:
        actual_a, actual_b = function()
        difference = max(
            float(np.max(np.abs(actual_a - reference_a))),
            float(np.max(np.abs(actual_b - reference_b))),
        )
        print(name, format_time(measure(function)), f"maxdiff={difference:.2f}")

    image_size = (size, size)

    print("\n-- цветовая доля --")
    presence_backends = [("presence_numpy ", df._color_presence_custom)]
    if df.CV2_AVAILABLE:
        presence_backends.append(("presence_cv    ", df._color_presence_cv))
    for name, function in presence_backends:
        print(name, format_time(
            measure(lambda function=function: function(
                reference_a, reference_b
            ))
        ))

    print("\n-- региональные средние, два канала --")
    for level in (2, 4, 8):
        if df.CV2_AVAILABLE:
            print(f"cv     level={level}", format_time(
                measure(
                    lambda level=level: df._color_region_means_cv(
                        reference_a, reference_b, level, image_size
                    )
                )
            ))
        print(f"custom level={level}", format_time(
            measure(
                lambda level=level: df._color_region_means_custom(
                    reference_a, reference_b, level, image_size
                )
            )
        ))
        if df.VIPS_AVAILABLE:
            print(f"vips   level={level}", format_time(
                measure(
                    lambda level=level: df._color_region_means_vips(
                        reference_a, reference_b, level, image_size
                    )
                )
            ))

    print("\n-- интеграл одного канала (для яркости/контраста/текстуры) --")
    if df.CV2_AVAILABLE:
        print("integral_cv    ", format_time(
            measure(lambda: df._integral_image_cv(reference_a, sq=False))
        ))
    print("integral_custom", format_time(
        measure(lambda: df._integral_image_custom(reference_a, sq=False))
    ))

    print("\n-- цветовой вектор, NumPy --")
    for level in (2, 4, 8):
        print(f"level={level}         ", format_time(
            measure(
                lambda level=level: df._calculate_color_vector(
                    reference_a, reference_b, level, image_size
                )
            )
        ))

    print("\n-- цветовой конвейер целиком --")
    for profile in ("cv", "vips", "pil"):
        print(f"pipeline {profile:4s}   ", format_time(
            measure(
                lambda profile=profile: color_pipeline(
                    profile, path, image_size
                ),
                repeat=5,
            )
        ))

    if df.CUPY_AVAILABLE and df._is_gpu_available():
        benchmark_gpu(image, image_size)


def benchmark_gpu(
    image: np.ndarray,
    image_size: tuple[int, int],
) -> None:
    print("\n-- GPU: отдельные операции --")
    image_gpu = df.to_gpu(image)
    lab_a, lab_b = df._get_ab_from_BGR_gpu(image_gpu)

    print("to_gpu         ", format_time(
        measure(lambda: df.to_gpu(image), gpu=True)
    ))
    print("gauss_gpu      ", format_time(
        measure(lambda: df._gauss_blur_gpu(image_gpu, (3, 3)), gpu=True)
    ))
    print("lab_gpu        ", format_time(
        measure(lambda: df._get_ab_from_BGR_gpu(image_gpu), gpu=True)
    ))
    print("presence_gpu   ", format_time(
        measure(lambda: df._color_presence(lab_a, lab_b), gpu=True)
    ))

    print("\n-- GPU: региональные средние --")
    for level in (2, 4, 8):
        print(f"level={level}         ", format_time(
            measure(
                lambda level=level: df._color_region_means_gpu(
                    lab_a, lab_b, level, image_size
                ),
                gpu=True,
            )
        ))

    print("\n-- GPU: цветовой вектор --")
    for level in (2, 4, 8):
        print(f"level={level}         ", format_time(
            measure(
                lambda level=level: df._calculate_color_vector(
                    lab_a, lab_b, level, image_size
                ),
                gpu=True,
            )
        ))

    print("\n-- GPU: конвейер целиком (без декодирования) --")
    print("pipeline gpu   ", format_time(
        measure(
            lambda: color_pipeline_gpu(image, image_size),
            repeat=5,
            gpu=True,
        )
    ))


def main() -> None:
    print_header()
    with tempfile.TemporaryDirectory() as temp_dir:
        for size in (1024, 2048):
            benchmark_size(size, Path(temp_dir))


if __name__ == "__main__":
    main()
