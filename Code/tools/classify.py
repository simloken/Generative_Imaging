import torch
from torch import nn, optim
import os
from tools.data import MNISTDataset
from torch.utils.data import DataLoader


class Classifier(nn.Module):
    """
    Convolutional Neural Network classifier for MNIST.

    Architecture:
      - Two convolutional layers with ReLU activations and max pooling.
      - Followed by two fully connected layers to produce logits for 10 classes.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        """
        Forward pass through the classifier.

        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, 1, 28, 28].

        Returns:
            torch.Tensor: Logits for each of the 10 classes.
        """
        return self.model(x)


# Global classifier instance to avoid reloading.
_classifier = None


def load_classifier(filepath="../Data/classifier.pth"):
    """
    Loads the pre-trained classifier model from disk.

    If the classifier has not been loaded yet, initializes the model,
    loads state from the specified file, and sets it to evaluation mode.

    Args:
        filepath (str): Path to the saved classifier model file.

    Returns:
        Classifier: The loaded and evaluated classifier model.

    Raises:
        FileNotFoundError: If the model file is not found at the specified filepath.
    """
    global _classifier
    if _classifier is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _classifier = Classifier().to(device)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pre-trained classifier not found at {filepath}.")
        _classifier.load_state_dict(torch.load(filepath, map_location=device))
        _classifier.eval()
    return _classifier


def classify_number(image, filepath="../Data/classifier.pth", confidence_threshold=0.995):
    """
    Classifies a single MNIST-like image and returns its predicted label.

    Converts the input image to a 4D tensor if needed, performs inference,
    and applies a confidence threshold to determine if the prediction is reliable.

    Args:
        image (torch.Tensor): A 28x28 grayscale image tensor.
        filepath (str): Path to the saved classifier model.
        confidence_threshold (float): Minimum confidence required for a valid prediction.

    Returns:
        Union[int, str]: Predicted digit (0-9) if confidence exceeds the threshold;
                         otherwise, returns "Uncertain".
    """
    classifier = load_classifier(filepath)
    device = next(classifier.parameters()).device
    with torch.no_grad():
        # Ensure the image has batch and channel dimensions.
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        logits = classifier(image)
        probabilities = torch.softmax(logits, dim=1)
        max_prob, predicted_label = torch.max(probabilities, dim=1)

        if max_prob.item() < confidence_threshold:
            return "Uncertain"
        return predicted_label.item()


def generate_classifier(data_path="../Data/mnist_train.csv", filepath="../Data/classifier.pth", epochs=10):
    """
    Trains a new MNIST classifier and saves it, if no pre-trained classifier exists.

    Loads the MNIST training data from a CSV file, initializes a classifier,
    trains the model using cross-entropy loss and the Adam optimizer,
    and then saves the trained model to disk.

    Args:
        data_path (str): Path to the CSV file containing MNIST training data.
        filepath (str): Filepath to save the trained classifier.
        epochs (int): Number of training epochs.

    Returns:
        None. Prints training progress and final evaluation accuracy.
    """
    if os.path.exists(filepath):
        print(f"Classifier already exists at {filepath}.")
        return

    print("Training classifier...")
    # Initialize dataset and dataloader.
    train_dataset = MNISTDataset(data_path, skip=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize classifier, optimizer, and loss function.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    # Training loop.
    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.unsqueeze(1).to(device), labels.to(device)  # Add channel dimension.
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Save the trained classifier.
    torch.save(classifier.state_dict(), filepath)
    print(f"Classifier with {evaluate_classifier()}% accuracy saved at {filepath}.")


def evaluate_classifier(test_data_path="../Data/mnist_test.csv", filepath="../Data/classifier.pth"):
    """
    Evaluates the pre-trained classifier on the MNIST test dataset.

    Loads the test data from a CSV file, runs inference using the classifier,
    and computes the overall accuracy.

    Args:
        test_data_path (str): Path to the CSV file containing MNIST test data.
        filepath (str): Path to the saved classifier model.

    Returns:
        float: Accuracy of the classifier as a fraction between 0 and 1.
    """
    test_dataset = MNISTDataset(test_data_path, skip=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    classifier = load_classifier(filepath)
    device = next(classifier.parameters()).device

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.unsqueeze(1).to(device), labels.to(device)  # Add channel dimension.
            outputs = classifier(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy
