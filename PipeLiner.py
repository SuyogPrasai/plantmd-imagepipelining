from pathlib import Path
from typing import List
import cv2
import numpy as np

class PipeLiner:
    def __init__(self, input_folder: Path, output_folder: Path):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.image_width = 200
        self.image_height = 200
        self.image_contrast = 10
        self.image_brightness = 10
        self.image_noise = 1

        self.output_folder.mkdir(parents=True, exist_ok=True)

    def initialize_images(self) -> List[Path]:
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder '{self.input_folder}' does not exist.")
        if not self.input_folder.is_dir():
            raise NotADirectoryError(f"'{self.input_folder}' is not a directory.")
        
        return [p for p in self.input_folder.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

    def read_image(self, image_path: Path) -> np.ndarray:
        return cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        height, width = image.shape[:2]
        rect = (10, 10, width - 20, height - 20)
        
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return image * mask2[:, :, np.newaxis]

    def process_image(self, image_path: Path) -> None:
        image = self.read_image(image_path)
        contrast_image = self.enhance_contrast(image)
        result_image = self.remove_background(contrast_image)
        output_path = self.output_folder / f"{image_path.stem}_output.jpg"
        cv2.imwrite(str(output_path), result_image)
        print(f"Processed: {image_path.name} -> {output_path.name}")

    def process_all_images(self) -> None:
        image_paths = self.initialize_images()
        for image_path in image_paths:
            self.process_image(image_path)