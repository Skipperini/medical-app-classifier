import cv2
import numpy as np
from PIL import Image
import io

MIN_THRESHOLD = 0.6


def average_hash(cv2_img, hash_size=20):
    img = cv2.resize(cv2_img, (hash_size, hash_size))
    binary = (img > np.mean(img)).astype(int)
    return binary.flatten()


def average_hash_from_file(img_path, hash_size=20):
    img = cv2.resize(
        cv2.imread(img_path, cv2.IMREAD_GRAYSCALE),
        (hash_size, hash_size)
    )
    binary = (img > np.mean(img)).astype(int)
    return binary.flatten()


def compare_hashes(hash1, hash2):
    return np.sum(hash1 == hash2) / len(hash1)


mri_templates = (
    average_hash_from_file("assets/validation/mri_back.jpg"),
    average_hash_from_file("assets/validation/mri_top.jpg"),
    average_hash_from_file("assets/validation/mri_top_eyes.jpg"),
    average_hash_from_file("assets/validation/mri_side.jpg"),
)


async def validate_image_size(file) -> bool:
    image = Image.open(io.BytesIO(file))
    return min(image.size) > 128


async def validate_image_similarity(file) -> bool:
    img = cv2.imdecode(
        np.frombuffer(file, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    compares = []
    for template_hash in mri_templates:
        compares.append(compare_hashes(
            average_hash(img),
            template_hash)
        )
    return MIN_THRESHOLD <= max(compares)


async def validate_image(file) -> bool:
    return await validate_image_size(file) and await validate_image_similarity(file)
