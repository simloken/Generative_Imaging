import importlib
from tools.data import MNISTDataset, get_data_make_dataset
from torch.utils.data import DataLoader
from tools.usebest import use_best
from tools.classify import generate_classifier
import os

class Runner:
    """
    Runner class for training generative models on the MNIST dataset.

    This class serves as a flexible wrapper to initialize and train different 
    generative models (e.g., GAN, VAE, RBM, DDPM) using a unified interface.
    The appropriate model is dynamically loaded based on the method name provided 
    in the configuration.

    Attributes:
        config (dict): Configuration dictionary containing training settings and method name.
        method_name (str): Name of the generative method (e.g., 'GAN', 'RBM').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loading.
        data_path (str): Path to the MNIST data file.
        generate_labels (list): Optional list of labels to use for conditional generation.
        model (object): The instantiated generative model.
        data_loader (DataLoader): PyTorch DataLoader for the MNIST dataset.
    """

    def __init__(self, config):
        """
        Initializes the Runner instance.

        Args:
            config (dict): A dictionary containing the configuration for training. 
                           Expected keys include:
                               - "method" (str): Name of the generative method module.
                               - "epochs" (int): Number of epochs to train.
                               - "batch_size" (int): Size of each training batch.
                               - "data_path" (str): Path to the MNIST dataset.
                               - "use_best" (bool): Whether to override config with best known hyperparameters.
                               - "generate_labels" (list, optional): Labels to conditionally generate.
        """
        self.config = config
        self.method_name = config["method"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.data_path = config["data_path"]
        self.generate_labels = config.get("generate_labels", [])

        # Dynamically import the specified generative model module
        method_module = importlib.import_module(self.method_name)

        # Optionally override config with best known hyperparameters
        if self.config['use_best'] == True:
            self.config = use_best(self.method_name)
        
        
        train_path = '../Data/mnist_train.csv'
        test_path = '../Data/mnist_test.csv'
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print('Missing dataset - Starting download to ../Data/')
            get_data_make_dataset()
        
        classifier_path = "../Data/classifier.pth"
        
        # Ensure classifier is generated if not already present
        if not os.path.exists(classifier_path):
            print("Classifier not found. Training a new classifier...")
            generate_classifier(data_path="../Data/mnist_train.csv", filepath=classifier_path, epochs=10)
        else:
            print(f"Using existing classifier at {classifier_path}.")
        
        # Initialize the model from the imported module
        self.model = method_module.Model(self.config)

        # Load dataset
        skip = (self.method_name == 'RBM')
        dataset = MNISTDataset(self.data_path, skip)
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def run(self):
        """
        Trains the generative model using the specified configuration.

        Calls the model's `train` method with the loaded data and number of epochs.
        """
        self.model.train(self.data_loader, self.epochs)
