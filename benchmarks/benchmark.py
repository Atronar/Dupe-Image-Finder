"""Замеры быстродействия основных операций библиотеки.

Скрипт запускается вручную:

    python benchmarks/benchmark.py

Печатается медианное время нескольких повторов. Абсолютные значения
зависят от машины, поэтому интерес прежде всего представляет сравнение
запусков до и после изменений.
"""
from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

# При запуске из корня проекта Python добавляет в sys.path каталог benchmarks,
# а не корень, поэтому путь к библиотеке добавляем явно.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duplicatefinder as df


def measure(function: Callable[[], object], repeat: int = 5, number: int = 1) -> float:
    """Возвращает медианное время выполнения в секундах."""
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        for _ in range(number):
            function()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def format_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} мкс"
    return f"{seconds * 1e3:.2f} мс"


def make_image(path: Path, size: int) -> None:
    """Создаёт цветное изображение с шумом, чтобы не получить вырожденные данные."""
    cv2 = __import__("cv2")
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("не удалось записать тестовое изображение")


def main() -> None:
    size = 1024
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "benchmark.png"
        make_image(image_path, size)

        gray = np.random.default_rng(1).random((size, size), dtype=np.float32) * 255
        color_vectors = [np.random.default_rng(i).random(30, dtype=np.float32) for i in range(2)]

        results: list[tuple[str, float]] = []

        if df.CV2_AVAILABLE:
            results.append(("open_image_cv", measure(lambda: df.open_image_cv(str(image_path)))))
            results.append(("_integral_image_cv", measure(lambda: df._integral_image(gray, sq=True))))
            results.append(("_gauss_blur_cv", measure(lambda: df._gauss_blur(gray, (3, 3)))))
        if df.PIL_AVAILABLE:
            results.append(("open_image_pil", measure(lambda: df.open_image_pil(str(image_path)))))
        if df.VIPS_AVAILABLE:
            results.append(("open_image_vips", measure(lambda: df.open_image_vips(str(image_path)))))
        if not df.CV2_AVAILABLE:
            results.append(("_integral_image_custom", measure(lambda: df._integral_image(gray, sq=True))))
            results.append(("_gauss_blur_custom", measure(lambda: df._gauss_blur(gray, (3, 3)))))
        else:
            # Собственные реализации сравниваем напрямую, даже если они не выбраны.
            results.append((
                "_integral_image_custom",
                measure(lambda: df._integral_image_custom(gray, sq=True)),
            ))
            results.append((
                "_gauss_blur_custom",
                measure(lambda: df._gauss_blur_custom(gray, (3, 3)), repeat=3),
            ))

        for threads in (False, True):
            results.append((
                f"intensities level=4 threads={threads}",
                measure(
                    lambda threads=threads: df.intensities(
                        str(image_path),
                        partition_level=4,
                        use_threads=threads,
                        use_gpu=False,
                    ),
                    repeat=3,
                ),
            ))

        results.append((
            "similarity x10000",
            measure(lambda: df.similarity(color_vectors[0], color_vectors[1]), repeat=3, number=10000),
        ))

        print(f"Python: {sys.version.split()[0]}")
        print(f"NumPy: {np.__version__}, CuPy: {df.CUPY_AVAILABLE}, "
              f"OpenCV: {df.CV2_AVAILABLE}, pyvips: {df.VIPS_AVAILABLE}, "
              f"Pillow: {df.PIL_AVAILABLE}")
        print(f"Размер изображения: {size}x{size}")
        for name, seconds in results:
            print(f"{name:38s} {format_time(seconds)}")


if __name__ == "__main__":
    main()
