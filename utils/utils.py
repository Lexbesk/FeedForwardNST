import os

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from torchvision import datasets
import torch
import matplotlib.pyplot as plt
# import git


IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN_255 = np.array([123.675, 116.28, 103.53])
# Usually when normalizing 0..255 images only mean-normalization is performed -> that's why standard dev is all 1s here
IMAGENET_STD_NEUTRAL = np.array([1, 1, 1])


def load_image(img_path, target_width=None, target_height=None):
    img = cv.imread(img_path)[:, :, ::-1]
    if target_height is not None and target_width is None:
        h, w = img.shape[:2]
        target_width = int(target_height * (w / h))
        img = cv.resize(img, (target_width, target_height), interpolation=cv.INTER_CUBIC)
    elif target_width is not None and target_height is None:
        h, w = img.shape[:2]
        target_height = int(target_width * (h / w))
        img = cv.resize(img, (target_width, target_height), interpolation=cv.INTER_CUBIC)
    elif target_width is not None and target_height is not None:
        img = cv.resize(img, (target_width, target_height), interpolation=cv.INTER_CUBIC)

    img = img.astype(np.float32)
    img /= 255.0
    return img


def prepare_img_to_stylization(img_path, target_width, target_height, device, batch_size=4):
    img = load_image(img_path, target_width, target_height)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
    ])
    img = transform(img).to(device)
    img = img.repeat(batch_size, 1, 1, 1)
    return img


def get_training_metadata(training_config):
    # num_of_datapoints = training_config['subset_size'] * training_config['num_of_epochs']
    training_metadata = {
        # "commit_hash": git.Repo(search_parent_directories=True).head.object.hexsha,
        "content_weight": training_config['content_weight'],
        "style_weight": training_config['style_weight'],
        "tv_weight": training_config['tv_weight'],
        # "num_of_datapoints": num_of_datapoints
    }
    return training_metadata


def store_and_play_img(img, output_path):
    pass


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def post_process(img):
    img = img.numpy()[0]
    mean = np.resize(IMAGENET_MEAN_1, (3, 1, 1))
    std = np.resize(IMAGENET_STD_1, (3, 1, 1))
    img = img * std + mean
    img = (np.clip(img, 0., 1.) * 255).astype(np.uint8)
    img = np.moveaxis(img, 0, 2)
    return img[:, :, :]


def save(img, path, name):
    cv.imwrite(os.path.join(path, name), img)


def get_data_loader(config):
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.CenterCrop(config['image_size']),
        transforms.ToTensor(),  # 255 -> 1
        transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1),
    ])
    dataset = datasets.ImageFolder(config['dataset_path'], transform)
    dataLoader = DataLoader(dataset, batch_size=config['batch_size'], drop_last=True)
    return dataLoader


def total_variation(img_batch):
    batch_size = img_batch.shape[0]
    return (torch.sum(torch.abs(img_batch[:, :, :, :-1] - img_batch[:, :, :, 1:])) +
            torch.sum(torch.abs(img_batch[:, :, :-1, :] - img_batch[:, :, 1:, :]))) / batch_size


# dataset_path = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data', 'mscoco')
# loader = get_data_loader({'image_size': 256, 'dataset_path': dataset_path, 'batch_size': 5})
# for i, (p, _) in enumerate(loader):
#     if i == 2:
#         print(p.shape)
#         p1 = post_process(p)
#         print(p1.shape)
#         plt.imshow(p1)
#         plt.show()
#         break



