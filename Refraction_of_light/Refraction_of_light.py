import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS
import os
import gc
from typing import Dict, List, Tuple


class LightRefractionAnalyzer:
    """
    Класс для анализа преломления света на границах сред.
    Работает с произвольным количеством наборов изображений (минимум 2) с разными выдержками.
    Все вычисления производятся в линейном цветовом пространстве (после sRGB линеаризации).
    """
    
    def __init__(self, directory_path: str, media_names: List[str], radius: int = 350):
        if len(media_names) < 2:
            raise ValueError("Требуется минимум 2 названия сред.")
        
        self.directory_path = directory_path
        self.radius = radius
        self.media_names = media_names
        self.reference_medium = self.media_names[0]
        self.comparison_media = self.media_names[1:]
        self.image_sets = {}
        self.exposure_times = []
        self.pixel_amplitudes = {}
        self.mean_amplitudes = {}
        
    def _get_exposure_time(self, image_path: str) -> float:
        """Извлекает время выдержки из EXIF. Возвращает значение в секундах (float)."""
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

    def _float_to_fraction_label(self, exp_time: float) -> str:
        if exp_time is None or exp_time <= 0:
            return "N/A"
        if exp_time >= 1:
            return f"{int(exp_time)}"
        else:
            denominator = round(1 / exp_time)
            if abs(1/denominator - exp_time) < 1e-6:
                return f"1/{denominator}"
            else:
                return f"{exp_time:.5g}"

    def _linearize_srgb(self, srgb_values: np.ndarray) -> np.ndarray:
        """
        Преобразует значения из нелинейного sRGB в линейное пространство.
        
        :param srgb_values: Массив значений в диапазоне [0, 255] (тип float)
        :return: Массив линейных значений
        """
        # Нормализуем в диапазон [0, 1]
        normalized = srgb_values / 255.0
        
        # Применяем обратную sRGB гамма-коррекцию
        # Ветвление для значений <= 0.04045 и > 0.04045
        linear = np.where(
            normalized <= 0.04045,
            normalized / 12.92,
            np.power((normalized + 0.055) / 1.055, 2.4)
        )
        
        # Возвращаем в исходный масштаб для удобства (или можно оставить в [0,1])
        # Для анализа отношений масштаб не важен, но для удобства отладки оставим как float.
        return linear

    def _load_images_from_directories(self):
        """Загружает изображения из поддиректорий."""
        for medium in self.media_names:
            dir_path = os.path.join(self.directory_path, medium)
            if not os.path.exists(dir_path):
                raise ValueError(f"Директория {dir_path} не существует")
            
            image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff'))]
            image_paths = [os.path.join(dir_path, f) for f in image_files]
            
            exp_groups = {}
            for path in image_paths:
                exp_time = self._get_exposure_time(path)
                if exp_time is not None:
                    key = round(exp_time, 6)
                    if key not in exp_groups:
                        exp_groups[key] = []
                    exp_groups[key].append(path)
            
            self.image_sets[medium] = exp_groups
            self.exposure_times.extend(list(exp_groups.keys()))
        
        self.exposure_times = sorted(list(set(self.exposure_times)))

    def _extract_blue_channel(self, image_path: str) -> np.ndarray:
        """
        Извлекает синий канал изображения и применяет sRGB линеаризацию.
        Возвращает линейные значения как float.
        """
        image = Image.open(image_path).convert("RGB")
        blue_channel = np.array(image)[:, :, 2].astype(np.float32)
        # Применяем линеаризацию
        linear_blue = self._linearize_srgb(blue_channel)
        return linear_blue

    def _center_of_mass(self, img: np.ndarray) -> Tuple[float, float]:
        y_coords, x_coords = np.indices(img.shape)
        total_intensity = np.sum(img)
        if total_intensity == 0:
            return img.shape[0] / 2, img.shape[1] / 2
        x_com = np.sum(img * x_coords) / total_intensity
        y_com = np.sum(img * y_coords) / total_intensity
        return y_com, x_com

    def _create_circular_mask(self, shape: tuple, center: Tuple[float, float], radius: int) -> np.ndarray:
        y_coords, x_coords = np.indices(shape)
        dist_from_center = np.sqrt((y_coords - center[0])**2 + (x_coords - center[1])**2)
        mask = dist_from_center <= radius
        return mask

    def prepare_data(self):
        """Подготавливает данные: загружает изображения и вычисляет амплитуды в линейном пространстве."""
        self._load_images_from_directories()
        
        for medium in self.media_names:
            self.pixel_amplitudes[medium] = {}
            self.mean_amplitudes[medium] = {}
            
            for exp_time, image_paths in self.image_sets[medium].items():
                if len(image_paths) == 0:
                    continue
                img = self._extract_blue_channel(image_paths[0])
                self.pixel_amplitudes[medium][exp_time] = img
                self.mean_amplitudes[medium][exp_time] = np.mean(img)

    def _plot_ratio_vs_exposure(self, exposures, ratios_dict, title, filename, output_dir):
        x_labels = [self._float_to_fraction_label(et) for et in exposures]
        x_positions = range(len(exposures))
        
        plt.figure(figsize=(10, 6))
        
        for medium_name, ratios in ratios_dict.items():
            label = f'{medium_name} / {self.reference_medium}'
            plt.plot(x_positions, ratios, 'o-', label=label, alpha=0.8, linewidth=2, markersize=6)
        
        plt.xticks(x_positions, x_labels, rotation=45)
        plt.xlabel('Exposure Time')
        plt.ylabel('Amplitude Ratio')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


    def method1_full_image(self, output_dir: str = "./results/method1/"):
        os.makedirs(output_dir, exist_ok=True)
        
        ratios_dict = {medium: [] for medium in self.comparison_media}
        exposures = []
        
        for exp_time in self.exposure_times:
            if all(exp_time in self.mean_amplitudes[medium] for medium in self.media_names):
                ref_val = self.mean_amplitudes[self.reference_medium][exp_time]
                if ref_val > 0:
                    for medium in self.comparison_media:
                        comp_val = self.mean_amplitudes[medium][exp_time]
                        ratios_dict[medium].append(comp_val / ref_val)
                    exposures.append(exp_time)
        
        self._plot_ratio_vs_exposure(
            exposures,
            ratios_dict,
            'Mean Amplitude Ratios vs Exposure Time (Full Image)',
            'mean_ratios_vs_exposure.png',
            output_dir
        )
        
        for exp_time in self.exposure_times:
            if all(exp_time in self.pixel_amplitudes[medium] for medium in self.media_names):
                ref_img = self.pixel_amplitudes[self.reference_medium][exp_time]
                ref_mask = ref_img > 0
                
                for medium in self.comparison_media:
                    comp_img = self.pixel_amplitudes[medium][exp_time]
                    comp_ratios = np.zeros_like(comp_img)
                    comp_ratios[ref_mask] = comp_img[ref_mask] / ref_img[ref_mask]
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(comp_ratios.flatten(), bins=100, alpha=0.7, label=f'{medium}/{self.reference_medium}', density=True)
                    plt.xlabel('Amplitude Ratio')
                    plt.ylabel('Density')
                    plt.title(f'Pixel-wise Ratios ({medium}/{self.reference_medium})\nExp. time: {self._float_to_fraction_label(exp_time)}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'pixel_ratios_{medium}_{exp_time:.6f}.png'))
                    plt.close()
        
        del ratios_dict, exposures
        gc.collect()


    def method2_center_of_mass(self, output_dir: str = "./results/method2/"):
        os.makedirs(output_dir, exist_ok=True)
        
        ratios_dict = {medium: [] for medium in self.comparison_media}
        exposures = []
        
        for exp_time in self.exposure_times:
            if all(exp_time in self.pixel_amplitudes[medium] for medium in self.media_names):
                ref_img = self.pixel_amplitudes[self.reference_medium][exp_time]
                com_y, com_x = self._center_of_mass(ref_img)
                mask = self._create_circular_mask(ref_img.shape, (com_y, com_x), self.radius)
                
                ref_roi = ref_img[mask]
                ref_mean = np.mean(ref_roi) if len(ref_roi) > 0 else 0
                
                if ref_mean > 0:
                    for medium in self.comparison_media:
                        comp_roi = self.pixel_amplitudes[medium][exp_time][mask]
                        comp_mean = np.mean(comp_roi) if len(comp_roi) > 0 else 0
                        ratios_dict[medium].append(comp_mean / ref_mean)
                    exposures.append(exp_time)
        
        self._plot_ratio_vs_exposure(
            exposures,
            ratios_dict,
            'Mean Amplitude Ratios (Center of Mass ROI) vs Exposure Time',
            'mean_ratios_roi_vs_exposure.png',
            output_dir
        )
        
        for exp_time in self.exposure_times:
            if all(exp_time in self.pixel_amplitudes[medium] for medium in self.media_names):
                ref_img = self.pixel_amplitudes[self.reference_medium][exp_time]
                com_y, com_x = self._center_of_mass(ref_img)
                mask = self._create_circular_mask(ref_img.shape, (com_y, com_x), self.radius)
                
                ref_roi = ref_img[mask]
                ref_mask_valid = ref_roi > 0
                
                for medium in self.comparison_media:
                    comp_roi = self.pixel_amplitudes[medium][exp_time][mask]
                    comp_ratios = np.zeros_like(comp_roi)
                    comp_ratios[ref_mask_valid] = comp_roi[ref_mask_valid] / ref_roi[ref_mask_valid]
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(comp_ratios, bins=100, alpha=0.7, label=f'{medium}/{self.reference_medium} (ROI)', density=True)
                    plt.xlabel('Amplitude Ratio')
                    plt.ylabel('Density')
                    plt.title(f'Pixel-wise Ratios ({medium}/{self.reference_medium} ROI)\nExp. time: {self._float_to_fraction_label(exp_time)}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'pixel_ratios_roi_{medium}_{exp_time:.6f}.png'))
                    plt.close()
        
        del ratios_dict, exposures
        gc.collect()


    def method3_combined_images(self, output_dir: str = "./results/method3/"):
        os.makedirs(output_dir, exist_ok=True)
        
        averaged_images = {}
        averaged_means = {}
        
        for medium in self.media_names:
            averaged_images[medium] = {}
            averaged_means[medium] = {}
            
            for exp_time, image_paths in self.image_sets[medium].items():
                if len(image_paths) == 0:
                    continue
                
                combined_img = None
                for path in image_paths:
                    img = self._extract_blue_channel(path) # Уже линейный!
                    if combined_img is None:
                        combined_img = img.astype(np.float64)
                    else:
                        combined_img += img
                
                if combined_img is not None:
                    averaged_img = combined_img / len(image_paths)
                    averaged_images[medium][exp_time] = averaged_img
                    averaged_means[medium][exp_time] = np.mean(averaged_img)
        
        ratios_dict = {medium: [] for medium in self.comparison_media}
        exposures = []
        
        for exp_time in self.exposure_times:
            if all(exp_time in averaged_means[medium] for medium in self.media_names):
                ref_val = averaged_means[self.reference_medium][exp_time]
                if ref_val > 0:
                    for medium in self.comparison_media:
                        comp_val = averaged_means[medium][exp_time]
                        ratios_dict[medium].append(comp_val / ref_val)
                    exposures.append(exp_time)
        
        self._plot_ratio_vs_exposure(
            exposures,
            ratios_dict,
            'Mean Amplitude Ratios (Combined Images) vs Exposure Time',
            'mean_ratios_combined_vs_exposure.png',
            output_dir
        )
        
        for exp_time in self.exposure_times:
            if all(exp_time in averaged_images[medium] for medium in self.media_names):
                ref_combined = averaged_images[self.reference_medium][exp_time]
                ref_mask = ref_combined > 0
                
                for medium in self.comparison_media:
                    comp_combined = averaged_images[medium][exp_time]
                    comp_ratios = np.zeros_like(comp_combined)
                    comp_ratios[ref_mask] = comp_combined[ref_mask] / ref_combined[ref_mask]
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(comp_ratios.flatten(), bins=100, alpha=0.7, label=f'{medium}/{self.reference_medium} (Combined)', density=True)
                    plt.xlabel('Amplitude Ratio')
                    plt.ylabel('Density')
                    plt.title(f'Pixel-wise Ratios ({medium}/{self.reference_medium} Combined)\nExp. time: {self._float_to_fraction_label(exp_time)}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'pixel_ratios_combined_{medium}_{exp_time:.6f}.png'))
                    plt.close()
        
        del ratios_dict, exposures, averaged_images, averaged_means
        gc.collect()


    def run_all_methods(self):
        print("Подготовка данных...")
        self.prepare_data()
        
        print("Запуск метода 1 (всё изображение)...")
        self.method1_full_image()
        
        print("Запуск метода 2 (центр масс)...")
        self.method2_center_of_mass()
        
        print("Запуск метода 3 (усреднённые изображения)...")
        self.method3_combined_images()
        
        print("Анализ завершён. Результаты сохранены в соответствующих поддиректориях.")
    
    def cleanup(self):
        if hasattr(self, 'pixel_amplitudes'):
            del self.pixel_amplitudes
        if hasattr(self, 'mean_amplitudes'):
            del self.mean_amplitudes
        if hasattr(self, 'image_sets'):
            del self.image_sets
        if hasattr(self, 'exposure_times'):
            del self.exposure_times
        gc.collect()


if __name__ == "__main__":
    analyzer = LightRefractionAnalyzer(
        directory_path="refraction_of_light/",
        media_names=['without_glass', 'with_1_glass', 'with_2_glass'],
        radius=750
    )
    analyzer.run_all_methods()
    analyzer.cleanup()
    del analyzer
    gc.collect()