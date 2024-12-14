import os
import torch
import torchvision
import numpy as np
from scipy.io import loadmat
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader

from LeNet5_2 import LeNet5


def load_data_from_mat(directory):
    images_all = []
    labels_all = []

    for i in range(1, 33):
        mat_path = os.path.join(directory, f"{i}.mat")
        mat_data = loadmat(mat_path)
        aff_data = mat_data['affNISTdata']
        images = aff_data['image'][0, 0]
        labels = aff_data['label_int'][0, 0]
        images = images.T
        images = images.reshape(-1, 40, 40)
        labels = labels.squeeze()

        images_all.append(images)
        labels_all.append(labels)

    images_all = np.concatenate(images_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    return images_all, labels_all


def preprocess_data(images, labels):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    processed_images = []
    for img in images:
        img_pil = Image.fromarray(img.astype(np.uint8))
        img_tensor = transform(img_pil)
        processed_images.append(img_tensor)

    processed_images = torch.stack(processed_images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return processed_images, labels_tensor


def test(dataloader, model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            outputs = model(data)
            _, predicted = torch.max(-outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_accuracy = correct / total
    print("test accuracy:", test_accuracy)


def main():
    directory = "training_and_validation_batches"
    images, labels = load_data_from_mat(directory)
    test_images, test_labels = preprocess_data(images, labels)

    test_dataset = TensorDataset(test_images, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = LeNet5()
    state_dict = torch.load("LeNet1.pth")
    model.load_state_dict(state_dict)
    model.eval()

    test(test_dataloader, model)


if __name__ == "__main__":
    main()
