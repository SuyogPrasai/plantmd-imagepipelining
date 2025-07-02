from PipeLiner import PipeLiner
from pathlib import Path

INPUT_DIR = Path("./input")
OUTPUT_DIR = Path("./output")

pipeliner = PipeLiner(INPUT_DIR, OUTPUT_DIR)

images = pipeliner.initialize_images()

for image in images:
    pipeliner.process_image(image)
