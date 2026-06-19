import torchvision.transforms as T


def train_transform():
    return T.Compose([
        T.ColorJitter(brightness=0.4, contrast=0.15, saturation=0.10, hue=0.02),
        T.RandomGrayscale(p=0.25),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def eval_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
