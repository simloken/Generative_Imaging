from runner import Runner
import os

if __name__ == "__main__":

    # Configuration for the generative model
    config = {
        "data_path": "../Data/mnist_train.csv",
        "epochs": 100,
        "batch_size": 100,
        "method": "GAN",
        "generate_labels": None,
        "use_best": False,
        "verbose": False,
    }
    runner = Runner(config)
    runner.run()
