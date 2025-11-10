from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from pathlib import Path

class LightAttenuationAnalyzer:
    """
    Класс для анализа затухания света в воде по сравнению с воздухом на основе фотографий.
    """
    def __init__(self, photos_list_files, output_dir="../Photos/Pictures/"):
        """
        Инициализирует анализатор.

        :param photos_list_files: Словарь с ключами 'air', 'water', 'dark' и путями к файлам со списками фото.
                                  Пример: {'air': 'air.txt', 'water': 'water.txt', 'dark': 'dark.txt'}
        :param output_dir: Директория для сохранения результатов.
        """
        self.photos_list_files = {k: Path(v) for k, v in photos_list_files.items()}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.photo_groups = self._load_photo_groups()
        self.exposure_groups = self._group_photos_by_exposure()

    def _load_photo_groups(self):
        """Загружает списки фото для воздуха, воды и темновых кадров."""
        groups = {}
        for key in ['air', 'water', 'dark']:
            file_path = self.photos_list_files.get(key)
            if file_path and file_path.exists():
                with open(file_path, "r", encoding='utf-8') as file:
                    groups[key] = [line.strip() for line in file if line.strip()]
            else:
                print(f"Файл для {key} не найден или не указан: {file_path}. Используется пустой список.")
                groups[key] = []
        return groups

    def _get_exposure_time(self, image_path):
        """Извлекает время выдержки из EXIF. Возвращает значение в секундах (float)."""
        try:
            image = Image.open(image_path)
            exif = image._getexif()
            if exif is not None:
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    if tag_name == "ExposureTime":
                        if isinstance(value, tuple) and len(value) == 2:
                            return value[0] / value[1]  # дробь: (1, 60) → 1/60
                        return float(value)
            return None
        except Exception as e:
            print(f"Ошибка при чтении EXIF {image_path}: {e}")
            return None

    def _group_photos_by_exposure(self):
        """Группирует фото по времени выдержки."""
        exposure_groups = {}
        for key in ['air', 'water', 'dark']: # Группируем все типы
            for path in self.photo_groups[key]:
                exp_time = self._get_exposure_time(path)
                if exp_time is not None:
                    key_exp = round(exp_time, 6)
                    if key_exp not in exposure_groups:
                        exposure_groups[key_exp] = {'air': [], 'water': [], 'dark': []}
                    exposure_groups[key_exp][key].append(path)
        return exposure_groups

    def _find_lambda(self, i_water, i_air, l_air, l):
        """Вычисляет длину ослабления."""
        valid = i_air != 0
        ratio = np.zeros_like(i_water, dtype=np.float32)
        ratio[valid] = i_water[valid] / i_air[valid]
        ratio = np.clip(ratio, a_min=1e-10, a_max=0.9999999)
        log_ratio = np.log(ratio)
        l_water = l / np.abs(log_ratio)
        l_water = np.nan_to_num(l_water, nan=0, posinf=0, neginf=0)
        return l_water

    def _remove_gradient(self, channel):
        """Удаляет градиент из канала."""
        def gradient_model(xy, a, b, c):
            x, y = xy
            return a * x + b * y + c

        x_coords, y_coords = np.indices(channel.shape)
        x_flat = x_coords.ravel()
        y_flat = y_coords.ravel()
        z_flat = channel.ravel()

        popt, _ = curve_fit(
            f=gradient_model,
            xdata=(x_flat, y_flat),
            ydata=z_flat,
            p0=[0, 0, np.mean(channel)]
        )

        a, b, c = popt
        gradient_background = gradient_model((x_coords, y_coords), *popt)
        cleaned = channel - gradient_background
        cleaned[cleaned < 0] = 0
        return cleaned, gradient_background

    def _plot_gradient_comparison(self, original, corrected, title, filename):
        """Сохраняет изображения до и после коррекции градиента."""
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='Blues', origin='upper')
        plt.colorbar()
        plt.title(f'До коррекции: {title}')

        plt.subplot(1, 2, 2)
        plt.imshow(corrected, cmap='Blues', origin='upper')
        plt.colorbar()
        plt.title(f'После коррекции: {title}')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def analyze(self, mode=0, radius=750, l_air_default=4000, l_default=2):
        """
        Основной метод для запуска анализа.

        :param mode: 0 - центральная область, 1 - всё изображение, 2 - усреднение по группам выдержки.
        :param radius: Радиус для режима 0.
        :param l_air_default: Значение l_air для режима 0.
        :param l_default: Значение l для режима 0.
        """
        self.mode = mode
        self.radius = radius
        self.l_air_default = l_air_default
        self.l_default = l_default

        if self.mode == 0:
            self._analyze_mode_0()
        elif self.mode == 1:
            self._analyze_mode_1()
        elif self.mode == 2:
            self._analyze_mode_2()
        else:
            print("Ошибка ввода: используйте 0, 1 или 2")
            return

    def _analyze_mode_0(self):
        """Анализ для центральной области."""
        # Цикл по группам выдержки
        for exp_time, group in self.exposure_groups.items():
            air_list = group['air']
            water_list = group['water']
            # Предполагаем, что в каждой группе есть как минимум одна пара
            min_len = min(len(air_list), len(water_list))
            for i in range(min_len):
                air_path = air_list[i]
                water_path = water_list[i]

                base_a = os.path.basename(air_path)
                base_w = os.path.basename(water_path)

                print(f"Режим 0 - Сравнение при выдержке {exp_time:.6f} с: {base_a} ↔ {base_w}")

                image_w = Image.open(water_path).convert("RGB")
                w_arr = np.array(image_w)
                w_blue = w_arr[:, :, 2].astype(np.float32)

                image_a = Image.open(air_path).convert("RGB")
                a_arr = np.array(image_a)
                a_blue = a_arr[:, :, 2].astype(np.float32)

                x_coord, y_coord = np.indices(w_blue.shape)
                x_center = w_blue.shape[0] / 2
                y_center = w_blue.shape[1] / 2
                center_vect = np.sqrt((x_coord - x_center)**2 + (y_coord - y_center)**2)
                mask = center_vect <= self.radius

                water_pix = np.where(mask, w_blue, 0)
                air_pix = np.where(mask, a_blue, 0)

                mask1 = np.where(water_pix > 0)
                water_pix1 = water_pix[mask1]
                air_pix1 = air_pix[mask1]

                if len(air_pix1) > 0 and np.mean(air_pix1) > 0:
                    mean_ratio = np.mean(water_pix1) / np.mean(air_pix1)
                    if mean_ratio > 0:
                        mean_lamb = 1 / (1/self.l_air_default - np.log(mean_ratio) / self.l_default)
                    else:
                        mean_lamb = np.nan
                else:
                    mean_lamb = np.nan

                L_att_water = self._find_lambda(water_pix, air_pix, self.l_air_default, self.l_default)
                hist_1 = L_att_water[(L_att_water > 0.1) & (L_att_water < 20)]

                print(f"Среднее λ (центр): {np.mean(hist_1):.2f}, точек: {len(hist_1)}")

                plt.hist(hist_1, bins=100, color='blue', alpha=0.7,
                         label=f'λ (среднее = {np.mean(hist_1):.2f})', range=[0, 20])
                plt.legend()
                plt.grid()
                plt.title(f"Распределение λ (центр, exp={exp_time:.6f})")
                plt.xlabel("Цветовой канал - синий")
                plt.savefig(self.output_dir / f"attenuation_dist0_{base_w}.png")
                plt.close()

                air_flat = air_pix[air_pix > 0]
                water_flat = water_pix[water_pix > 0]

                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.hist(air_flat, bins=256, color='blue', alpha=0.7, range=[0, 255])
                plt.grid()
                plt.title(f"Интенсивность в воздухе (центр, exp={exp_time:.6f})")
                plt.xlabel("Синий канал")

                plt.subplot(1, 2, 2)
                plt.hist(water_flat, bins=256, color='blue', alpha=0.7, range=[0, 255])
                plt.grid()
                plt.title(f"Интенсивность в воде (центр, exp={exp_time:.6f})")
                plt.xlabel("Синий канал")
                plt.tight_layout()
                plt.savefig(self.output_dir / f"intensity_dist0_{base_w}.png")
                plt.close()

                print(f"Распределение на попиксельной основе: {np.mean(hist_1):.2f}, "
                      f"на средней основе: {mean_lamb:.2f}")


    def _analyze_mode_1(self):
        """Анализ для всего изображения."""
        for exp_time, group in self.exposure_groups.items():
            air_list = group['air']
            water_list = group['water']
            min_len = min(len(air_list), len(water_list))
            for i in range(min_len):
                air_path = air_list[i]
                water_path = water_list[i]

                base_a = os.path.basename(air_path)
                base_w = os.path.basename(water_path)

                print(f"Режим 1 - Сравнение при выдержке {exp_time:.6f} с: {base_a} ↔ {base_w}")

                image_w = Image.open(water_path).convert("RGB")
                w_arr = np.array(image_w)
                w_blue = w_arr[:, :, 2].astype(np.float32)

                image_a = Image.open(air_path).convert("RGB")
                a_arr = np.array(image_a)
                a_blue = a_arr[:, :, 2].astype(np.float32)

                L_att_water = self._find_lambda(w_blue, a_blue, 10000, 0.82)
                L_att_water = L_att_water[(L_att_water > 1) & (L_att_water < 30)]

                plt.hist(L_att_water, bins=100, color='blue', alpha=0.7,
                         label=f'λ (среднее = {np.mean(L_att_water):.2f})', range=[0, 30])
                plt.legend()
                plt.grid()
                plt.title(f"Распределение λ (всё, exp={exp_time:.6f})")
                plt.xlabel("Синий канал")
                plt.savefig(self.output_dir / f"attenuation_dist1_{base_w}.png")
                plt.close()

                a_flat = a_blue.flatten()
                w_flat = w_blue.flatten()

                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.hist(a_flat, bins=256, color='blue', alpha=0.7, range=[0, 255])
                plt.grid()
                plt.title(f"Интенсивность в воздухе (exp={exp_time:.6f})")
                plt.xlabel("Синий канал")

                plt.subplot(1, 2, 2)
                plt.hist(w_flat, bins=256, color='blue', alpha=0.7, range=[0, 255])
                plt.grid()
                plt.title(f"Интенсивность в воде (exp={exp_time:.6f})")
                plt.xlabel("Синий канал")
                plt.tight_layout()
                plt.savefig(self.output_dir / f"intensity_dist1_{base_w}.png")
                plt.close()

    def _analyze_mode_2(self):
        """
        Режим 2: Группировка по выдержке. Для каждой группы:
        1. Усреднить синие каналы фото в воздухе и фото в воде.
        2. Вычислить среднюю длину ослабления.
        """
        print("Запуск анализа в режиме 2: Усреднение по группам выдержки.")
        results = []

        for exp_time, group in self.exposure_groups.items():
            air_list = group['air']
            water_list = group['water']
            # Темновые кадры пока не используются, но доступны: group['dark']

            if not air_list or not water_list:
                print(f"Пропуск группы с выдержкой {exp_time:.6f}: нет фото воздуха или воды.")
                continue

            print(f"Обработка группы с выдержкой {exp_time:.6f}: {len(air_list)} air, {len(water_list)} water")

            # Загрузка и усреднение каналов воздуха
            avg_air_blue = np.zeros_like(Image.open(air_list[0]).convert("RGB")[:, :, 2], dtype=np.float64)
            for path in air_list:
                img = Image.open(path).convert("RGB")
                arr = np.array(img)
                avg_air_blue += arr[:, :, 2].astype(np.float64)
            avg_air_blue /= len(air_list)

            # Загрузка и усреднение каналов воды
            avg_water_blue = np.zeros_like(Image.open(water_list[0]).convert("RGB")[:, :, 2], dtype=np.float64)
            for path in water_list:
                img = Image.open(path).convert("RGB")
                arr = np.array(img)
                avg_water_blue += arr[:, :, 2].astype(np.float64)
            avg_water_blue /= len(water_list)

            # Вычисление средней длины ослабления
            lambda_att = self._find_lambda(avg_water_blue, avg_air_blue, self.l_air_default, self.l_default)
            # Фильтруем для статистики
            valid_lambdas = lambda_att[(lambda_att > 0.1) & (lambda_att < 20)]
            mean_lambda = np.mean(valid_lambdas) if len(valid_lambdas) > 0 else np.nan

            results.append({
                'exp_time': exp_time,
                'avg_air_intensity': np.mean(avg_air_blue),
                'avg_water_intensity': np.mean(avg_water_blue),
                'mean_att_length': mean_lambda,
                'valid_pixels_count': len(valid_lambdas)
            })

            print(f"  - Средняя инт. воздуха: {results[-1]['avg_air_intensity']:.2f}")
            print(f"  - Средняя инт. воды: {results[-1]['avg_water_intensity']:.2f}")
            print(f"  - Средняя длина ослабления: {mean_lambda:.2f} (на {results[-1]['valid_pixels_count']} пикселях)")

    # --- Новый универсальный метод ---
    def plot_attenuation_vs_exposure(self, data_x, data_y, title_suffix="", xlabel="Время выдержки (с)", ylabel="Средняя длина ослабления λ (условные ед.)", plot_filename="attenuation_vs_exposure.png"):
        """
        Строит график зависимости средней длины ослабления от времени выдержки.

        :param data_x: Список или массив значений по оси X (например, время выдержки).
                       Или список словарей, тогда используется ключ x_key.
        :param data_y: Список или массив значений по оси Y (например, средние длины ослабления).
                       Или ключ (str), если data_x - список словарей.
        :param title_suffix: Дополнительная строка для заголовка графика (опционально).
        :param xlabel: Название оси X.
        :param ylabel: Название оси Y.
        :param plot_filename: Имя файла для сохранения графика.
        """
        # Проверяем, являются ли оба аргумента списками/массивами
        if isinstance(data_x, (list, tuple)) and isinstance(data_y, (list, tuple)):
            x_vals = data_x
            y_vals = data_y
        else:
            print("Неподдерживаемый формат данных. Ожидается два списка/массива.")
            return

        if len(x_vals) != len(y_vals):
            print("Ошибка: количество значений X и Y должно совпадать.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, 'o-', label=ylabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs {xlabel}{title_suffix}')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / plot_filename)
        plt.close()
        print(f"График зависимости длины ослабления от выдержки сохранён: {self.output_dir / plot_filename}")