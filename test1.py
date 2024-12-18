import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from data import df_test  
from PIL import Image
import io


def preprocess_data(df):

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    images = []
    labels = []
    for row in df.itertuples():
        img = Image.open(io.BytesIO(row.image["bytes"]))
        img_tensor = transform(img)
        images.append(img_tensor)
        labels.append(row.label)
    return torch.stack(images), torch.tensor(labels)


def test(dataloader, model):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            outputs = model(data)
            _, predicted = torch.max(-outputs, -1) 
            #print(f"Batch {batch_idx}: Predicted: {predicted}, Target: {target}")  # Debugging
            total += target.size(0)
            correct += (predicted == target).sum().item()
    test_accuracy = correct / total if total > 0 else 0
    print(f"Total: {total}, Correct: {correct}")  
    print("Test Accuracy:", test_accuracy * 100)

def main():
    test_images, test_labels = preprocess_data(df_test)
    test_dataset = TensorDataset(test_images, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    from LeNet5_1_new import LeNet5 
    model = LeNet5()

    model.load_state_dict(torch.load("LeNet1.pth"))
    model.eval()

    test(test_dataloader, model)

if __name__ == "__main__":
    main()
