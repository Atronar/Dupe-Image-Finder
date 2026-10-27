Dupe Image Finder
==================

**Dupe Image Finder** — библиотека для сравнения изображений по набору
векторов: средней яркости, контрасту и текстуре. Алгоритм основан на
анализе изображения и его четвертей (логика
[этого кода на Ruby](https://gist.github.com/liamwhite/b023cdba4738e911293a8c610b98f987)).

Требования
----------

* Python >= 3.10;
* NumPy;
* OpenCV устанавливается как базовая зависимость;
* Pillow, pyvips и CuPy — необязательные ускорения и альтернативные декодеры.

Установка
---------

```bash
pip install .
```

Дополнительные возможности:

```bash
pip install ".[pillow]"   # альтернативный декодер
pip install ".[vips]"     # libvips
pip install ".[gpu]"      # CuPy для NVIDIA GPU
pip install ".[dev]"      # pytest
```

CuPy должен соответствовать установленной версии CUDA. Если CUDA
недоступна, библиотека автоматически продолжит работу на CPU.

Быстрый старт
-------------

```python
import duplicatefinder as df

vectors = df.intensities("image.jpg")
print(len(vectors))       # число характеристик
print(vectors[0].shape)   # размерность вектора яркости
```

`intensities()` возвращает список векторов. При `partition_level=2`
каждый вектор имеет длину `1² + 2² = 5`. Первый вектор описывает
среднюю яркость, второй — контраст, третий — текстуру.

Основные параметры
------------------

* `partition_level` — уровень разбиения изображения; размерность каждого
  вектора равна `sum(n²)` для `n` от 1 до `partition_level`;
* `blur_ksize` — размер ядра размытия Гаусса; `(0, 0)` отключает размытие;
* `use_gpu` — попытаться использовать CuPy; при отсутствии CUDA
  вычисления автоматически выполняются на CPU;
* `use_threads` — параллельный расчёт характеристик одного изображения.
  По умолчанию отключён: после векторизации уровней потоки для одного
  изображения дают только накладные расходы. Для набора изображений
  используйте `intensities_iter()`.

Приоритет бэкендов
------------------

Для загрузки, размытия и интегральных изображений выбирается самый
быстрый доступный вариант:

1. CuPy (GPU);
2. OpenCV;
3. pyvips;
4. собственная реализация на NumPy.

Если не установлен ни один декодер, библиотека всё равно импортируется,
но для чтения изображений потребуется OpenCV, Pillow или pyvips.

Методы
------

[intensities(image_path)](#intensities)<br>
[intensities_iter(image_list)](#intensities_iter)<br>
[similarity(vector1, vector2)](#similarity)<br>
[is_similar(vector1, vector2, threshold=0.1)](#is_similar)<br>
[similarity_images(image1, image2)](#similarity_images)<br>
[is_similar_images(image1, image2, threshold=0.1)](#is_similar_images)<br>

#### intensities

**intensities(image_path: str|Path|BinaryIO, partition_level=2,
blur_ksize=(3, 3), use_threads=False, use_gpu=True) -> list[array]**<br>
Главная функция модуля. Возвращает список векторов, соответствующих
изображению.

Изображение приводится к BGR `float32` в диапазоне `0…255`. Учитывается
EXIF-ориентация, 16-битные изображения нормализуются, альфа-канал
композитируется на белом фоне. Для GIF читается первый кадр.

#### intensities_iter

**intensities_iter(image_list: Iterable[str|Path|BinaryIO], **kwargs) ->
Generator[list[array]]**<br>
Обёртка над [intensities](#intensities) для набора файлов. Изображения
обрабатываются параллельно, порядок результатов сохраняется. Подходит
для построения базы векторов с последующим поиском дубликатов.

#### similarity

**similarity(vector1: array, vector2: array) -> float ∈ [0; 1]**<br>
Уровень похожести между двумя векторами.<br>
0 — векторы идентичны<br>
1 — векторы максимально разные

Векторы должны быть одномерными, непустыми и конечными.

#### is_similar

**is_similar(vector1: array, vector2: array, threshold: float=0.1) -> bool**<br>
Возвращает *True*, если векторы похожи. Порог должен быть конечным
неотрицательным числом.

#### similarity_images

**similarity_images(image1: str|Path|BinaryIO, image2: str|Path|BinaryIO)
-> list[float]**<br>
Сравнивает изображения и возвращает список расстояний по каждой
характеристике.

#### is_similar_images

**is_similar_images(image1: str|Path|BinaryIO, image2: str|Path|BinaryIO,
threshold: float=0.1) -> bool**<br>
Возвращает *True*, если изображения похожи по всем характеристикам.

Тесты и бенчмарк
----------------

```bash
pip install -r requirements-dev.txt
python -m pytest -q --disable-warnings
python benchmarks/benchmark.py
```

Бенчмарк печатает медианное время загрузки, интегральных изображений,
размытия, расчёта характеристик на разных уровнях разбиения и сравнения
векторов.
