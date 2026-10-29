from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_BLOCK_IMPORTS = r"""
import sys

class _Blocker:
    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.names:
            raise ImportError("заблокирован: " + fullname)

sys.meta_path.insert(0, _Blocker({blocked!r}))
sys.path.insert(0, {repo!r})
import duplicatefinder
"""


def _run_without_modules(blocked: tuple[str, ...], body: str = "") -> subprocess.CompletedProcess:
    code = _BLOCK_IMPORTS.format(blocked=blocked, repo=str(REPO_ROOT))
    if body:
        code += "\n" + body
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def _write_color_png(path: Path, size: tuple[int, int] = (32, 32)) -> Path:
    cv2 = pytest.importorskip("cv2")
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[..., 0] = 10  # B
    image[..., 1] = 20  # G
    image[..., 2] = 30  # R
    assert cv2.imwrite(str(path), image)
    return path


def _write_gray_png(path: Path, size: tuple[int, int] = (32, 32)) -> Path:
    cv2 = pytest.importorskip("cv2")
    image = np.full((size[1], size[0]), 128, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def _write_color_shape_png(path: Path) -> Path:
    """Цветное изображение с крупной цветной областью."""
    cv2 = pytest.importorskip("cv2")
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    image[24:104, 24:104] = (0, 0, 255)  # красный квадрат
    assert cv2.imwrite(str(path), image)
    return path


def _write_gray_shape_png(path: Path) -> Path:
    cv2 = pytest.importorskip("cv2")
    image = np.full((128, 128), 255, dtype=np.uint8)
    image[24:104, 24:104] = 76
    assert cv2.imwrite(str(path), image)
    return path


def _write_gray_watermark_png(path: Path) -> Path:
    """Серое изображение с маленькой насыщенной цветной меткой."""
    cv2 = pytest.importorskip("cv2")
    image = np.full((128, 128, 3), 128, dtype=np.uint8)
    image[58:70, 58:70] = (0, 0, 255)  # около 0.9% площади
    assert cv2.imwrite(str(path), image)
    return path


def _write_pseudogray_png(path: Path) -> Path:
    """Трёхканальный файл, визуально серый из-за слабого цветового шума."""
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    gray = np.full((128, 128, 3), 128, dtype=np.int16)
    gray += rng.integers(-2, 3, gray.shape, dtype=np.int16)
    image = np.clip(gray, 0, 255).astype(np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


@pytest.fixture
def color_png(tmp_path: Path) -> Path:
    return _write_color_png(tmp_path / "color.png")


@pytest.fixture
def gray_png(tmp_path: Path) -> Path:
    return _write_gray_png(tmp_path / "gray.png")


@pytest.fixture
def large_color_png(tmp_path: Path) -> Path:
    return _write_color_png(tmp_path / "large.png", size=(600, 600))


@pytest.fixture
def gpu_color_png(tmp_path: Path) -> Path:
    # Больше порога 2 Мпикс, чтобы GPU действительно включился.
    return _write_color_png(tmp_path / "gpu.png", size=(1536, 1536))


def _vectors(result) -> list[np.ndarray]:
    vectors = [np.asarray(vector) for vector in result]
    assert vectors, "библиотека вернула пустой набор компонент"
    return vectors


def _assert_vectors(result, partition_level: int = 2) -> None:
    expected_size = sum(level ** 2 for level in range(1, partition_level + 1))
    vectors = _vectors(result)
    assert len(vectors) == 4
    for vector in vectors[:3]:
        assert vector.ndim == 1
        assert vector.size == expected_size
        assert np.isfinite(vector).all()

    color = vectors[-1]
    assert color.ndim == 1
    assert color.size == 5 * expected_size
    assert np.isfinite(color).all()
    assert np.all(color >= 0.0)
    assert np.all(color <= 1.0)


@pytest.mark.parametrize(
    "blocked",
    [
        ("cupy",),
        ("cv2",),
        ("cupy", "cv2", "pyvips", "PIL"),
    ],
)
def test_import_without_optional_dependencies(blocked: tuple[str, ...]) -> None:
    """Импорт не должен требовать CuPy, OpenCV или другой декодер изображений."""
    result = _run_without_modules(blocked)
    assert result.returncode == 0, result.stderr


def test_runtime_without_cupy(large_color_png: Path) -> None:
    """При отсутствии CuPy обработка изображения должна продолжиться на CPU."""
    pytest.importorskip("cv2")
    body = (
        f"duplicatefinder.intensities({str(large_color_png)!r}, "
        "partition_level=1, use_threads=False, use_gpu=True)\n"
        "print('ok')"
    )
    result = _run_without_modules(("cupy",), body)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize("blocked", [("cv2",), ("cv2", "pyvips")])
def test_core_operations_without_opencv(
    df,
    large_color_png: Path,
    blocked: tuple[str, ...],
) -> None:
    """Без OpenCV загрузка и базовые операции идут через pyvips/Pillow/NumPy."""
    if not (df.VIPS_AVAILABLE or df.PIL_AVAILABLE):
        pytest.skip("нет ни pyvips, ни Pillow")
    if "pyvips" in blocked:
        pytest.importorskip("PIL")

    body = (
        "from duplicatefinder import open_image, _gauss_blur, _integral_image\n"
        f"image = open_image({str(large_color_png)!r})\n"
        "blurred = _gauss_blur(image, (3, 3))\n"
        "integral, integral_sq = _integral_image(blurred, sq=True)\n"
        "print(integral.shape, integral_sq.shape)\n"
    )
    result = _run_without_modules(blocked, body)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("blocked", [("cv2",), ("cv2", "pyvips")])
def test_intensities_without_opencv(
    df,
    large_color_png: Path,
    blocked: tuple[str, ...],
) -> None:
    """Цветовая характеристика должна считаться и без OpenCV."""
    if "pyvips" in blocked:
        pytest.importorskip("PIL")
    body = (
        "import duplicatefinder as library\n"
        f"vectors = library.intensities({str(large_color_png)!r}, "
        "partition_level=1, use_threads=False, use_gpu=False)\n"
        "print(len(vectors), vectors[-1].shape)\n"
    )
    result = _run_without_modules(blocked, body)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "4 (5,)"


def test_gpu_probe_handles_unavailable_cuda(df, monkeypatch) -> None:
    """Импортируемый CuPy без рабочей CUDA не должен приводить к падению."""
    if not df.CUPY_AVAILABLE:
        pytest.skip("CuPy не установлен")

    monkeypatch.setattr(df, "_gpu_available", None)

    def fail() -> int:
        raise RuntimeError("CUDA недоступна")

    monkeypatch.setattr(df.cp.cuda.runtime, "getDeviceCount", fail)

    assert df._is_gpu_available() is False
    array = np.zeros((2, 2), dtype=df.num_dtype)
    assert df.to_gpu(array) is array


def test_gpu_probe_handles_missing_cuda_headers(df, monkeypatch) -> None:
    """Устройство может быть доступно, но JIT-компиляция ядер — нет."""
    if not df.CUPY_AVAILABLE:
        pytest.skip("CuPy не установлен")

    monkeypatch.setattr(df, "_gpu_available", None)
    monkeypatch.setattr(df.cp.cuda.runtime, "getDeviceCount", lambda: 1)

    def fail(*args, **kwargs):
        raise RuntimeError("не найдены заголовки CUDA")

    monkeypatch.setattr(df.cp, "arange", fail)

    assert df._is_gpu_available() is False


def test_intensities_falls_back_to_cpu_without_cuda(df, color_png: Path, monkeypatch) -> None:
    """При недоступной CUDA обработка продолжается на CPU."""
    monkeypatch.setattr(df, "_gpu_available", False)
    result = df.intensities(
        str(color_png),
        partition_level=1,
        use_threads=False,
        use_gpu=True,
    )
    _assert_vectors(result, partition_level=1)


def test_intensities_gray_numpy2(df, gray_png: Path) -> None:
    """Серое изображение должно обрабатываться на NumPy 2 без ошибок копирования."""
    result = df.intensities(
        str(gray_png),
        partition_level=2,
        use_threads=False,
        use_gpu=False,
    )
    _assert_vectors(result)


def test_intensities_color(df, color_png: Path) -> None:
    result = df.intensities(
        str(color_png),
        partition_level=2,
        use_threads=False,
        use_gpu=False,
    )
    _assert_vectors(result)


def test_color_distance_distinguishes_color_and_gray(df, tmp_path: Path) -> None:
    """Цветное и серое изображения должны различаться цветовой компонентой."""
    color_path = _write_color_shape_png(tmp_path / "color-shape.png")
    gray_path = _write_gray_shape_png(tmp_path / "gray-shape.png")

    color_vectors = df.intensities(
        str(color_path), partition_level=2, use_threads=False, use_gpu=False
    )
    gray_vectors = df.intensities(
        str(gray_path), partition_level=2, use_threads=False, use_gpu=False
    )

    distance = df.similarity(color_vectors[-1], gray_vectors[-1])
    assert distance > 0.5
    assert df.is_similar_images(
        str(color_path), str(gray_path),
        partition_level=2, use_threads=False, use_gpu=False,
    ) is False


def test_similarity_images_uses_similarity_for_all_vectors(
    df, color_png: Path, gray_png: Path
) -> None:
    """similarity_images должен оставаться обычным сравнением всех векторов."""
    vectors1 = df.intensities(
        str(color_png), partition_level=2, use_threads=False, use_gpu=False
    )
    vectors2 = df.intensities(
        str(gray_png), partition_level=2, use_threads=False, use_gpu=False
    )
    expected = [
        df.similarity(first, second)
        for first, second in zip(vectors1, vectors2)
    ]
    actual = df.similarity_images(
        str(color_png), str(gray_png),
        partition_level=2, use_threads=False, use_gpu=False,
    )
    np.testing.assert_allclose(actual, expected)


def test_color_distance_ignores_small_watermark(df, tmp_path: Path) -> None:
    """Мелкая цветная метка не должна делать серое изображение цветным."""
    gray_path = _write_gray_shape_png(tmp_path / "gray.png")
    watermark_path = _write_gray_watermark_png(tmp_path / "watermark.png")

    gray_vectors = df.intensities(
        str(gray_path), partition_level=2, use_threads=False, use_gpu=False
    )
    watermark_vectors = df.intensities(
        str(watermark_path), partition_level=2, use_threads=False, use_gpu=False
    )

    assert df.similarity(gray_vectors[-1], watermark_vectors[-1]) < 0.1


def test_pseudogray_is_neutral(df, tmp_path: Path) -> None:
    """Трёхканальный, но визуально серый файл не должен считаться цветным."""
    gray_path = _write_gray_shape_png(tmp_path / "gray.png")
    pseudo_path = _write_pseudogray_png(tmp_path / "pseudogray.png")

    gray_vectors = df.intensities(
        str(gray_path), partition_level=2, use_threads=False, use_gpu=False
    )
    pseudo_vectors = df.intensities(
        str(pseudo_path), partition_level=2, use_threads=False, use_gpu=False
    )

    assert df.similarity(gray_vectors[-1], pseudo_vectors[-1]) < 0.05


def test_ab_backends_agree(df, color_png: Path) -> None:
    """Все доступные реализации Lab должны давать близкие a и b."""
    image = df.open_image(str(color_png))
    reference_a, reference_b = df._get_ab_from_BGR_cv(image)
    backends = [("custom", df._get_ab_from_BGR_custom)]
    if df.VIPS_AVAILABLE:
        backends.append(("vips", df._get_ab_from_BGR_vips))
    if df._is_gpu_available():
        backends.append((
            "gpu",
            lambda value: tuple(
                df.from_gpu(component)
                for component in df._get_ab_from_BGR_gpu(df.to_gpu(value))
            ),
        ))

    for name, backend in backends:
        actual_a, actual_b = backend(image)
        # CPU-путь OpenCV использует uint8 Lab ради скорости, поэтому
        # допускаем погрешность квантования до трёх единиц Lab.
        np.testing.assert_allclose(
            actual_a, reference_a, atol=3.0,
            err_msg=f"канал a, бэкенд {name}",
        )
        np.testing.assert_allclose(
            actual_b, reference_b, atol=3.0,
            err_msg=f"канал b, бэкенд {name}",
        )


def test_intensities_iter_preserves_parameter_names(df, color_png: Path, gray_png: Path) -> None:
    """intensities_iter должен передавать параметры по именам, а не по порядку словаря."""
    images = [str(color_png), str(gray_png)]
    kwargs = {
        "use_gpu": False,
        "blur_ksize": (1, 1),
        "partition_level": 1,
        "use_threads": True,
    }
    actual = list(df.intensities_iter(images, **kwargs))
    expected_kwargs = {**kwargs, "use_threads": False}
    expected = [df.intensities(image, **expected_kwargs) for image in images]

    assert len(actual) == len(expected)
    for actual_vectors, expected_vectors in zip(actual, expected):
        assert len(actual_vectors) == len(expected_vectors)
        for actual_vector, expected_vector in zip(actual_vectors, expected_vectors):
            np.testing.assert_array_equal(actual_vector, expected_vector)


def test_image_decoders_agree_on_bgr(df, color_png: Path) -> None:
    """Все доступные декодеры должны возвращать одинаковый BGR-массив."""
    openers = []
    if df.CV2_AVAILABLE:
        openers.append(df.open_image_cv)
    if df.PIL_AVAILABLE:
        openers.append(df.open_image_pil)
    if df.VIPS_AVAILABLE:
        openers.append(df.open_image_vips)
    if len(openers) < 2:
        pytest.skip("нужно хотя бы два декодера")

    arrays = [opener(str(color_png)) for opener in openers]
    for array in arrays[1:]:
        assert array.shape == arrays[0].shape
        np.testing.assert_allclose(array, arrays[0], atol=1)
    # color_png записан OpenCV в BGR: B=10, G=20, R=30.
    np.testing.assert_allclose(arrays[0][0, 0], [10, 20, 30], atol=1)


def test_gpu_intensities_matches_cpu(df, gpu_color_png: Path) -> None:
    """GPU-путь должен давать те же характеристики, что и CPU."""
    if not df._is_gpu_available():
        pytest.skip("CUDA недоступна")

    cpu = df.intensities(
        str(gpu_color_png),
        partition_level=2,
        use_threads=False,
        use_gpu=False,
    )
    gpu = df.intensities(
        str(gpu_color_png),
        partition_level=2,
        use_threads=False,
        use_gpu=True,
    )
    assert len(cpu) == len(gpu)
    for index, (cpu_vector, gpu_vector) in enumerate(zip(cpu, gpu)):
        # На константном изображении контраст и текстура близки к нулю.
        # Разница порядка 0.01 — накопленная погрешность float64 при
        # вычитании близких больших сумм, а не смешивание каналов.
        # Цветовая доля на границе мягкого порога может отличаться
        # сильнее из-за квантования Lab в uint8 на CPU.
        atol = 0.1 if index == len(cpu) - 1 else 0.02
        np.testing.assert_allclose(gpu_vector, cpu_vector, rtol=1e-3, atol=atol)


def test_open_image_accepts_bytesio(df, color_png: Path) -> None:
    reference = df.open_image(str(color_png))
    actual = df.open_image(io.BytesIO(color_png.read_bytes()))
    np.testing.assert_allclose(actual, reference)


def test_open_image_accepts_raw_file(df, color_png: Path) -> None:
    """Файловый объект RawIOBase тоже должен считаться файлом, а не путём."""
    reference = df.open_image(str(color_png))
    with open(color_png, "rb", buffering=0) as image_file:
        actual = df.open_image(image_file)
    np.testing.assert_allclose(actual, reference)


def test_open_image_cv_accepts_raw_file(df, color_png: Path) -> None:
    if not getattr(df, "CV2_AVAILABLE", False):
        pytest.skip("OpenCV не установлен")
    reference = df.open_image_cv(str(color_png))
    with open(color_png, "rb", buffering=0) as image_file:
        actual = df.open_image_cv(image_file)
    np.testing.assert_allclose(actual, reference)


def test_open_image_vips_accepts_file_object(df, color_png: Path) -> None:
    if not getattr(df, "VIPS_AVAILABLE", False):
        pytest.skip("pyvips не установлен")
    reference = df.open_image_vips(str(color_png))
    actual = df.open_image_vips(io.BytesIO(color_png.read_bytes()), file_format="png")
    np.testing.assert_allclose(actual, reference)


def test_open_image_returns_bgr(df, color_png: Path) -> None:
    image = df.open_image(str(color_png))
    assert image.dtype == np.float32
    assert image.ndim == 3
    np.testing.assert_allclose(image[0, 0], [10, 20, 30], atol=1)


def test_open_image_cv_normalizes_uint16(df, tmp_path: Path) -> None:
    """16-битное изображение должно приходить в том же диапазоне 0…255."""
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "gray16.png"
    image = np.full((32, 32), 65535, dtype=np.uint16)
    assert cv2.imwrite(str(path), image)

    actual = df.open_image_cv(str(path))
    assert actual.dtype == np.float32
    assert actual.ndim == 2
    assert actual.max() == pytest.approx(255.0, abs=0.5)


def test_open_image_pil_returns_bgr_for_rgba(df, tmp_path: Path) -> None:
    """Полупрозрачный RGBA должен давать тот же BGR, что и OpenCV."""
    pil = pytest.importorskip("PIL.Image")
    path = tmp_path / "rgba.png"
    pil.new("RGBA", (8, 8), (10, 20, 30, 128)).save(path)

    actual = df.open_image_pil(str(path))
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual[0, 0], [142.06, 137.04, 132.02], atol=0.6)
    if getattr(df, "CV2_AVAILABLE", False):
        np.testing.assert_allclose(actual, df.open_image_cv(str(path)), atol=0.6)


def test_open_image_pil_handles_palette_image(df, tmp_path: Path) -> None:
    """Палитровое изображение не должно приниматься за серое."""
    pil = pytest.importorskip("PIL.Image")
    path = tmp_path / "palette.png"
    image = pil.new("P", (8, 8), 1)
    image.putpalette([0, 0, 0, 255, 0, 0])
    image.save(path)

    actual = df.open_image_pil(str(path))
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual[0, 0], [0.0, 0.0, 255.0], atol=0.5)


def test_open_image_pil_normalizes_uint16(df, tmp_path: Path) -> None:
    """Pillow для 16-битного PNG тоже должен вернуть диапазон 0…255."""
    pil = pytest.importorskip("PIL.Image")
    path = tmp_path / "gray16-pil.png"
    image = pil.new("I;16", (16, 16), 65535)
    image.save(path)

    actual = df.open_image_pil(str(path))
    assert actual.dtype == np.float32
    assert actual.max() == pytest.approx(255.0, abs=0.5)


def _save_jpeg_with_orientation(path: Path, orientation: int) -> None:
    pil = pytest.importorskip("PIL.Image")
    image = pil.new("RGB", (16, 8))
    pixels = image.load()
    for y in range(8):
        for x in range(16):
            pixels[x, y] = (255, 0, 0) if y < 4 else (0, 0, 255)
    exif = image.getexif()
    exif[274] = orientation
    # 4:4:4 без цветовой субдискретизации, чтобы OpenCV и Pillow
    # одинаково декодировали границу красной и синей областей.
    image.save(path, format="JPEG", exif=exif, quality=100, subsampling=0)


def test_open_image_pil_applies_exif_orientation(df, tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    _save_jpeg_with_orientation(path, 6)

    actual = df.open_image_pil(str(path))
    assert actual.shape == (16, 8, 3)


@pytest.mark.parametrize("orientation", range(2, 9))
def test_open_image_cv_applies_exif_orientation(df, tmp_path: Path, orientation: int) -> None:
    if not getattr(df, "CV2_AVAILABLE", False):
        pytest.skip("OpenCV не установлен")
    path = tmp_path / f"rotated-cv-{orientation}.jpg"
    _save_jpeg_with_orientation(path, orientation)

    actual = df.open_image_cv(str(path))
    expected = df.open_image_pil(str(path))
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=3.0)


def test_identical_vectors_are_similar(df, color_png: Path) -> None:
    vectors = df.intensities(
        str(color_png),
        partition_level=2,
        use_threads=False,
        use_gpu=False,
    )
    for vector in vectors:
        assert df.similarity(vector, vector) == pytest.approx(0.0, abs=1e-6)
        assert df.is_similar(vector, vector) is True


def test_integral_image_custom_uses_float64(df) -> None:
    """Собственный интеграл вещественного изображения не должен терять точность."""
    image = np.full((64, 64), 255, dtype=np.float32)

    integral, integral_sq = df._integral_image_custom(image, sq=True)

    assert integral.dtype == np.float64
    assert integral_sq.dtype == np.float64


def test_contrast_is_finite_for_large_uniform_image(
    df, tmp_path: Path, monkeypatch
) -> None:
    """На большом однородном изображении дисперсия не должна уходить в минус."""
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "uniform-large.png"
    image = np.full((2000, 2000), 255, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)

    # Принудительно проверяем собственную реализацию интеграла.
    monkeypatch.setattr(df, "CV2_AVAILABLE", False)
    monkeypatch.setattr(df, "VIPS_AVAILABLE", False)

    vectors = df.intensities(
        str(path),
        partition_level=1,
        blur_ksize=(0, 0),
        use_threads=False,
        use_gpu=False,
    )

    for vector in vectors:
        assert np.isfinite(vector).all()
    assert vectors[1][0] == pytest.approx(0.0, abs=1e-4)


def test_contrast_is_accurate_for_low_contrast_image(
    df, tmp_path: Path, monkeypatch
) -> None:
    """Низкоконтрастное изображение должно давать корректное стандартное отклонение."""
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "low-contrast-large.png"
    image = np.empty((2000, 2000), dtype=np.uint8)
    image[:, :1000] = 199
    image[:, 1000:] = 201
    assert cv2.imwrite(str(path), image)

    monkeypatch.setattr(df, "CV2_AVAILABLE", False)
    monkeypatch.setattr(df, "VIPS_AVAILABLE", False)

    vectors = df.intensities(
        str(path),
        partition_level=1,
        blur_ksize=(0, 0),
        use_threads=False,
        use_gpu=False,
    )

    assert vectors[1][0] == pytest.approx(1.0, abs=1e-3)


def test_rect_sum_returns_luminance_for_bgr(df) -> None:
    red = np.zeros((2, 2, 3), dtype=np.float32)
    red[..., 2] = 255.0

    assert df.rect_sum(red) == pytest.approx(54.23, abs=0.01)


def test_rect_sum_returns_intensity_for_grayscale(df) -> None:
    gray = np.full((2, 2), 255.0, dtype=np.float32)

    assert df.rect_sum(gray) == pytest.approx(255.0)


def test_rect_sum_composites_alpha_on_white(df) -> None:
    transparent_red = np.zeros((2, 2, 4), dtype=np.float32)
    transparent_red[..., 2] = 255.0
    transparent_red[..., 3] = 0.0

    assert df.rect_sum(transparent_red) == pytest.approx(255.0)

    transparent_red[..., 3] = 255.0
    assert df.rect_sum(transparent_red) == pytest.approx(54.23, abs=0.01)


@pytest.mark.parametrize("shape", [(32, 40), (32, 40, 3)])
def test_gauss_blur_custom_supports_non_square_kernel(df, shape) -> None:
    """Неквадратное ядро не должно менять форму изображения."""
    rng = np.random.default_rng(0)
    image = rng.random(shape, dtype=np.float32) * 255.0

    actual = df._gauss_blur_custom(image, (5, 3), sigma=1.2)

    assert actual.shape == image.shape
    assert actual.dtype == np.float32


def test_gauss_blur_custom_matches_opencv_for_non_square_kernel(df) -> None:
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    image = rng.random((32, 40, 3), dtype=np.float32) * 255.0

    actual = df._gauss_blur_custom(image, (5, 3), sigma=1.2)
    expected = cv2.GaussianBlur(image, (5, 3), 1.2)

    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_similarity_rejects_different_shapes(df) -> None:
    with pytest.raises(ValueError):
        df.similarity(np.zeros(3, dtype=np.float32), np.zeros(4, dtype=np.float32))


def test_similarity_rejects_non_finite_values(df) -> None:
    vector = np.array([0.0, 1.0, np.nan], dtype=np.float32)

    with pytest.raises(ValueError):
        df.similarity(vector, vector)


def test_similarity_rejects_empty_vectors(df) -> None:
    empty = np.array([], dtype=np.float32)

    with pytest.raises(ValueError):
        df.similarity(empty, empty)


def test_similarity_rejects_negative_threshold(df) -> None:
    vector = np.ones(3, dtype=np.float32)

    with pytest.raises(ValueError):
        df.is_similar(vector, vector, threshold=-0.1)


@pytest.mark.parametrize(
    ("size", "level"),
    [((10, 10), 3), ((13, 7), 4), ((1024, 768), 7)],
)
def test_generate_grid_coords_covers_whole_image(df, size, level) -> None:
    coords = list(df.generate_grid_coords(size[0], size[1], level))

    assert len(coords) == level * level
    x_intervals = [x for _, x in coords]
    y_intervals = [y for y, _ in coords]
    assert min(x0 for x0, _ in x_intervals) == 0
    assert max(x1 for _, x1 in x_intervals) == size[0]
    assert min(y0 for y0, _ in y_intervals) == 0
    assert max(y1 for _, y1 in y_intervals) == size[1]

    # Соседние регионы не должны пересекаться и оставлять промежутки.
    xs = sorted(set(x_intervals))
    ys = sorted(set(y_intervals))
    for (_, x1), (x0, _) in zip(xs, xs[1:]):
        assert x1 == x0
    for (_, y1), (y0, _) in zip(ys, ys[1:]):
        assert y1 == y0
