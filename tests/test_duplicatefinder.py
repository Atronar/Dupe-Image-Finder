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


@pytest.fixture
def color_png(tmp_path: Path) -> Path:
    return _write_color_png(tmp_path / "color.png")


@pytest.fixture
def gray_png(tmp_path: Path) -> Path:
    return _write_gray_png(tmp_path / "gray.png")


@pytest.fixture
def large_color_png(tmp_path: Path) -> Path:
    # Больше порога 512x512, чтобы библиотека не отключала GPU автоматически.
    return _write_color_png(tmp_path / "large.png", size=(600, 600))


def _vectors(result) -> list[np.ndarray]:
    vectors = [np.asarray(vector) for vector in result]
    assert vectors, "библиотека вернула пустой набор компонент"
    return vectors


def _assert_vectors(result, partition_level: int = 2) -> None:
    expected_size = sum(level ** 2 for level in range(1, partition_level + 1))
    for vector in _vectors(result):
        assert vector.ndim == 1
        assert vector.size == expected_size
        assert np.isfinite(vector).all()


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
