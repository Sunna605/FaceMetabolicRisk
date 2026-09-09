from torchvision import transforms


MEAN = (0.48145466, 0.45782750, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def _convert_to_rgb(image):
    return image.convert("RGB")


def eval_transform(image_size=224):
    return transforms.Compose(
        [
            _convert_to_rgb,
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def train_transform(image_size=224):
    return transforms.Compose(
        [
            _convert_to_rgb,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
            ),
            transforms.RandomRotation(degrees=25),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
