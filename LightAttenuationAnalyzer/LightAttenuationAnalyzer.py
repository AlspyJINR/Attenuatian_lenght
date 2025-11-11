from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

class LightAttenuationAnalyzer:
    def __init__(
        self,
        air_list_file,
        black_air_list_file,
        water_list_file,
        black_water_list_file,
        output_dir="../Photos/Pictures/03.09.25"
    ):
        """
        Инициализация с четырьмя отдельными файлами со списками путей.
        Каждый файл содержит построчно полные или относительные пути к изображениям.
        """
        self.files_config = {
            'air': air_list_file,
            'black_air': black_air_list_file,
            'water': water_list_file,
            'black_water': black_water_list_file
        }
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._load_photo_lists()
        self._group_photos_by_exposure()

    def _load_photo_lists(self):
        """Загружает списки фото из четырёх файлов."""
        self.photo_lists = {}
        for key, file_path in self.files_config.items():
            if os.path.exists(file_path):
                with open(file_path, "r", encoding='utf-8') as f:
                    paths = [line.strip() for line in f if line.strip()]
                self.photo_lists[key] = paths
                print(f"Загружено {len(paths)} фото для '{key}' из {file_path}")
            else:
                print(f"⚠️ Файл не найден: {file_path}. Используется пустой список.")
                self.photo_lists[key] = []

    def _get_exposure_time(self, image_path):
        try:
            image = Image.open(image_path)
            exif = image._getexif()
            if exif is not None:
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    if tag_name == "ExposureTime":
                        if isinstance(value, tuple) and len(value) == 2:
                            return value[0] / value[1]
                        return float(value)
            return None
        except Exception as e:
            print(f"Ошибка при чтении EXIF {image_path}: {e}")
            return None

    def _group_photos_by_exposure(self):
        """Группирует все фото по времени выдержки."""
        self.exposure_groups = {}

        # Обрабатываем все 4 типа
        for key in ['air', 'black_air', 'water', 'black_water']:
            for path in self.photo_lists[key]:
                exp_time = self._get_exposure_time(path)
                if exp_time is not None:
                    exp_key = round(exp_time, 6)
                    if exp_key not in self.exposure_groups:
                        self.exposure_groups[exp_key] = {
                            'air': [], 'black_air': [], 'water': [], 'black_water': []
                        }
                    self.exposure_groups[exp_key][key].append(path)
                else:
                    print(f"⚠️ Не удалось прочитать выдержку для {path} (тип: {key})")

        print(f"Создано {len(self.exposure_groups)} групп по выдержке.")

    def _load_blue_channel(self, path):
        """Загружает синий канал как float32 (0–255)."""
        return np.array(Image.open(path).convert("RGB"))[:, :, 2].astype(np.float32)

    def _linearize_srgb(self, v):
        """Линеаризация sRGB (v в [0,1]) → линейная яркость."""
        v = np.asarray(v, dtype=np.float64)
        return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

    def _center_of_mass(self, img):
        y_coords, x_coords = np.indices(img.shape)
        total = np.sum(img)
        if total == 0:
            return img.shape[0] / 2, img.shape[1] / 2
        x_com = np.sum(img * x_coords) / total
        y_com = np.sum(img * y_coords) / total
        return y_com, x_com

    def _find_lambda(self, i_water, i_air, l_air, l):
        valid = i_air > 1e-8
        ratio = np.full_like(i_water, np.nan, dtype=np.float64)
        ratio[valid] = (i_water[valid] * 0.84) / (i_air[valid] * 0.914)  # Поправки на отражения
        ratio = np.clip(ratio, 1e-10, 0.9999999)
        log_ratio = np.log(ratio)
        l_water = 1.0 / (1.0 / l_air - log_ratio / l)
        l_water = np.nan_to_num(l_water, nan=0.0, posinf=0.0, neginf=0.0)
        return l_water

    def calculate_uncertainties(
        self,
        lambda_values,
        I_water,
        I_air,
        L_path=0.82,
        L_air=4000.0,
        delta_L_rel=0.01,
        delta_f_fresnel=0.10,
        jpeg_sys_rel=0.07,
        include_fluorescence_sys=0.0
    ):
        """
        Рассчитывает статистическую и полную погрешность длины ослабления.
        
        Возвращает: (mean_lambda, stat_error, total_error)
        """
        if len(lambda_values) == 0:
            return np.nan, np.nan, np.nan

        # Статистическая ошибка
        mean_lambda = np.mean(lambda_values)
        stat_error = np.std(lambda_values) / np.sqrt(len(lambda_values))

        # Систематика через T
        T_raw = (np.mean(I_water) * 0.84) / (np.mean(I_air) * 0.914)
        T_raw = np.clip(T_raw, 1e-10, 0.9999999)

        lnT = np.log(T_raw)
        if lnT >= 0:
            total_error = np.nan
        else:
            # Относительная погрешность T
            delta_T_rel = np.sqrt(
                jpeg_sys_rel**2 +
                delta_f_fresnel**2 +
                include_fluorescence_sys**2
            )
            # Чувствительность λ к T
            dlambda_dT = (L_path / (T_raw * (1 - (L_path / L_air) * lnT)**2))
            # Абсолютная погрешность от T
            delta_lambda_T = np.abs(dlambda_dT) * T_raw * delta_T_rel
            # Погрешность от L_path и L_air
            delta_lambda_L = mean_lambda * delta_L_rel

            sys_error = np.sqrt(delta_lambda_T**2 + delta_lambda_L**2)
            total_error = np.sqrt(stat_error**2 + sys_error**2)

        return mean_lambda, stat_error, total_error

    def plot_histogram_with_errors(self, lambda_values, mean_lambda, total_error, filename, title_suffix=""):
        plt.figure(figsize=(8, 5))
        plt.hist(lambda_values, bins=100, color='steelblue', alpha=0.7, range=(0, np.percentile(lambda_values, 99) if len(lambda_values) > 0 else 20))
        if not np.isnan(mean_lambda):
            plt.axvline(mean_lambda, color='red', linestyle='--', label=f'Среднее = {mean_lambda:.2f}')
        if not np.isnan(total_error) and not np.isnan(mean_lambda):
            plt.axvspan(mean_lambda - total_error, mean_lambda + total_error, color='red', alpha=0.2, label=f'Погрешность = ±{total_error:.2f}')
        plt.xlabel("Длина ослабления λ")
        plt.ylabel("Частота")
        plt.title(f"Распределение λ {title_suffix}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def _analyze_mode_0(self, radius=350):
        for exp_time, group in self.exposure_groups.items():
            air_list = group['air']
            black_air_list = group['black_air']
            water_list = group['water']
            black_water_list = group['black_water']

            if not air_list or not water_list:
                continue

            min_len = min(len(air_list), len(water_list))
            for i in range(min_len):
                air_path = air_list[i]
                water_path = water_list[i]
                base_w = os.path.basename(water_path)

                try:
                    img_air = self._load_blue_channel(air_path)
                    img_water = self._load_blue_channel(water_path)
                except Exception as e:
                    print(f"Ошибка загрузки: {e}")
                    continue

                # === Темновая коррекция ===
                if black_air_list:
                    dark_air = np.mean([self._load_blue_channel(p) for p in black_air_list], axis=0)
                else:
                    dark_air = 0.0

                if black_water_list:
                    dark_water = np.mean([self._load_blue_channel(p) for p in black_water_list], axis=0)
                else:
                    dark_water = 0.0

                air_corr = np.clip(img_air - dark_air, 0, None)
                water_corr = np.clip(img_water - dark_water, 0, None)
                air_lin = self._linearize_srgb(air_corr / 255.0)
                water_lin = self._linearize_srgb(water_corr / 255.0)

                # === Центр масс ===
                y_com, x_com = self._center_of_mass(water_lin)
                h, w = water_lin.shape
                y_coords, x_coords = np.indices((h, w))
                dist = np.sqrt((y_coords - y_com)**2 + (x_coords - x_com)**2)
                mask = dist <= radius

                water_vals = water_lin[mask]
                air_vals = air_lin[mask]
                valid = air_vals > 1e-8
                if not np.any(valid):
                    print(f"Нет валидных пикселей: {base_w}")
                    continue

                water_valid = water_vals[valid]
                air_valid = air_vals[valid]

                lambda_map = self._find_lambda(water_valid, air_valid, l_air=4000.0, l=2.0)
                valid_lamb = lambda_map[(lambda_map > 0.1) & (lambda_map < 20)]

                mean_l, stat_err, total_err = self.calculate_uncertainties(
                    lambda_values=valid_lamb,
                    I_water=water_valid,
                    I_air=air_valid,
                    L_path=2.0,
                    L_air=4000.0,
                    delta_L_rel=0.01,
                    delta_f_fresnel=0.10,
                    jpeg_sys_rel=0.07,
                    include_fluorescence_sys=0.30
                )

                self.plot_histogram_with_errors(
                    lambda_values=valid_lamb,
                    mean_lambda=mean_l,
                    total_error=total_err,
                    filename=f"attenuation_with_errors_mode0_{base_w}.png",
                    title_suffix=f"(центр масс, exp={exp_time:.6f}s)"
                )

                # Визуализация маски
                plt.figure(figsize=(8, 8))
                plt.imshow(water_lin, cmap='Blues')
                plt.colorbar()
                plt.contour(mask, levels=[0.5], colors='red', linewidths=1)
                plt.plot(x_com, y_com, 'r+', markersize=12, mew=2, label='Центр масс')
                plt.title(f"Центр масс и маска (R={radius})\n{base_w}")
                plt.legend()
                plt.savefig(os.path.join(self.output_dir, f"mask_visualization_{base_w}.png"))
                plt.close()

                print(f"[Mode 0] {base_w}: λ = {mean_l:.3f} ± {total_err:.3f}")

    def _analyze_mode_1(self):
        for exp_time, group in self.exposure_groups.items():
            air_list = group['air']
            black_air_list = group['black_air']
            water_list = group['water']
            black_water_list = group['black_water']

            if not air_list or not water_list:
                continue

            min_len = min(len(air_list), len(water_list))
            for i in range(min_len):
                air_path = air_list[i]
                water_path = water_list[i]
                base_w = os.path.basename(water_path)

                try:
                    img_air = self._load_blue_channel(air_path)
                    img_water = self._load_blue_channel(water_path)
                except Exception as e:
                    print(f"Ошибка загрузки: {e}")
                    continue

                # === Темновая коррекция ===
                if black_air_list:
                    dark_air = np.mean([self._load_blue_channel(p) for p in black_air_list], axis=0)
                else:
                    dark_air = 0.0

                if black_water_list:
                    dark_water = np.mean([self._load_blue_channel(p) for p in black_water_list], axis=0)
                else:
                    dark_water = 0.0

                air_corr = np.clip(img_air - dark_air, 0, None)
                water_corr = np.clip(img_water - dark_water, 0, None)
                air_lin = self._linearize_srgb(air_corr / 255.0)
                water_lin = self._linearize_srgb(water_corr / 255.0)

                lambda_map = self._find_lambda(water_lin, air_lin, l_air=10000.0, l=0.82)
                valid_lamb = lambda_map[(lambda_map > 0.1) & (lambda_map < 30)]

                mean_l, stat_err, total_err = self.calculate_uncertainties(
                    lambda_values=valid_lamb,
                    I_water=water_lin.flatten(),
                    I_air=air_lin.flatten(),
                    L_path=0.82,
                    L_air=10000.0
                )

                self.plot_histogram_with_errors(
                    lambda_values=valid_lamb,
                    mean_lambda=mean_l,
                    total_error=total_err,
                    filename=f"attenuation_with_errors_mode1_{base_w}.png",
                    title_suffix=f"(всё изображение, exp={exp_time:.6f}s)"
                )

                print(f"[Mode 1] {base_w}: λ = {mean_l:.3f} ± {total_err:.3f}")

    def _analyze_mode_2(self):
        """
        Режим 2: Усреднение всех изображений в группе по выдержке.
        Для каждой группы:
          1. Усредняются темновые кадры (black_air, black_water).
          2. Усредняются основные кадры (air, water) с коррекцией темнового тока.
          3. Применяется линеаризация.
          4. Вычисляется карта λ и её статистика с погрешностями.
        """
        print("Запуск анализа в режиме 2: Усреднение по группам выдержки.")

        for exp_time, group in self.exposure_groups.items():
            air_list = group['air']
            water_list = group['water']
            black_air_list = group['black_air']
            black_water_list = group['black_water']

            if not air_list or not water_list:
                print(f"Пропуск группы с выдержкой {exp_time:.6f}: нет фото воздуха или воды.")
                continue

            print(f"Обработка группы с выдержкой {exp_time:.6f}: "
                  f"{len(air_list)} air, {len(water_list)} water, "
                  f"{len(black_air_list)} black_air, {len(black_water_list)} black_water")

            # === Усреднение темновых кадров ===
            if black_air_list:
                dark_air_stack = np.stack([self._load_blue_channel(p) for p in black_air_list], axis=0)
                dark_air_avg = np.mean(dark_air_stack, axis=0)
            else:
                dark_air_avg = 0.0

            if black_water_list:
                dark_water_stack = np.stack([self._load_blue_channel(p) for p in black_water_list], axis=0)
                dark_water_avg = np.mean(dark_water_stack, axis=0)
            else:
                dark_water_avg = 0.0

            # === Усреднение основных кадров ===
            air_stack = np.stack([self._load_blue_channel(p) for p in air_list], axis=0)
            water_stack = np.stack([self._load_blue_channel(p) for p in water_list], axis=0)
            air_avg = np.mean(air_stack, axis=0)
            water_avg = np.mean(water_stack, axis=0)

            # === Коррекция темнового тока и линеаризация ===
            air_corr = np.clip(air_avg - dark_air_avg, 0, None)
            water_corr = np.clip(water_avg - dark_water_avg, 0, None)
            air_lin = self._linearize_srgb(air_corr / 255.0)
            water_lin = self._linearize_srgb(water_corr / 255.0)

            # === Расчёт длины ослабления ===
            L_air = 4000.0  # эффективная длина для воздуха (условная)
            L_path = 2.0    # реальная длина трубы с водой

            lambda_map = self._find_lambda(water_lin, air_lin, l_air=L_air, l=L_path)
            valid_lambdas = lambda_map[(lambda_map > 0.1) & (lambda_map < 20)]

            if len(valid_lambdas) == 0:
                print(f"  → Нет валидных значений λ для выдержки {exp_time:.6f}")
                continue

            # === Расчёт погрешностей ===
            mean_l, stat_err, total_err = self.calculate_uncertainties(
                lambda_values=valid_lambdas,
                I_water=water_lin,
                I_air=air_lin,
                L_path=L_path,
                L_air=L_air,
                delta_L_rel=0.01,
                delta_f_fresnel=0.10,
                jpeg_sys_rel=0.07,
                include_fluorescence_sys=0.30
            )

            # === Сохранение гистограммы с ошибками ===
            self.plot_histogram_with_errors(
                lambda_values=valid_lambdas,
                mean_lambda=mean_l,
                total_error=total_err,
                filename=f"attenuation_mode2_exp_{exp_time:.6f}.png",
                title_suffix=f"(усреднение по группе, exp={exp_time:.6f}s)"
            )

            # === Вывод результатов ===
            print(f"  → Средняя длина ослабления: {mean_l:.3f} ± {total_err:.3f}")
            print(f"  → Использовано пикселей: {len(valid_lambdas)}")

    def analyze(self, mode=0, radius=350):
        """
        Запуск анализа.
        mode=0: центр масс,
        mode=1: всё изображение,
        mode=2: усреднение по группе выдержки.
        """
        if mode == 0:
            self._analyze_mode_0(radius=radius)
        elif mode == 1:
            self._analyze_mode_1()
        elif mode == 2:
            self._analyze_mode_2()
        else:
            raise ValueError("Режим должен быть 0, 1 или 2")