import os
os.environ["PYTORCH_NO_NNPACK"] = "1"

import argparse
from argparse import Namespace
import json
import os
import torch
import torch.nn.functional as F

from dataloader import MyECGDataset
from model import EnsembleECGModel  # Use EnsembleECGModel, ensuring it's taken from logs

if __name__ == "__main__":
    # System arguments
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument(
        "--input_data", type=str, default="data/test_data.h5", help="Path to H5 test data file."
    )
    sys_parser.add_argument(
        "--log_dir", type=str, default="logs/", help="Path to directory containing config and model."
    )
    settings, _ = sys_parser.parse_known_args()

    # Read config file from the logs folder
    config_path = os.path.join(os.getcwd(), settings.log_dir, "config.json")
    with open(config_path) as json_file:
        mydict = json.load(json_file)
    config = Namespace(**mydict)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------------
    # Create dataloader for the test data.
    # -----------------------------------------------------------------------------
    dataset = MyECGDataset(settings.input_data)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False
    )

    # -----------------------------------------------------------------------------
    # Initialize the model from logs folder
    # -----------------------------------------------------------------------------
    model = EnsembleECGModel(config, settings.log_dir)  # Model loaded from logs
    model.to(config.device)
    model.eval()  # Set model to evaluation mode

    # -----------------------------------------------------------------------------
    # Prediction loop.
    # -----------------------------------------------------------------------------
    class_labels = {0: "normal", 1: "stemi", 2: "nstemi"}

    for batch_idx, batch in enumerate(test_loader):
        # Extract data from batch.
        traces, labels, ids, age, sex = batch
        traces = traces.to(device=config.device)
        labels = labels.to(device=config.device)
        age_sex = torch.stack([sex, age]).t().to(device=config.device)

        # Forward pass.
        with torch.no_grad():
            inp = (traces, age_sex)
            logits = model(inp)
            probs = F.softmax(logits, dim=-1)

        # Print true and predicted labels with probabilities.
        for i in range(len(labels)):
            true_label = labels[i].item()
            predicted_label_index = torch.argmax(probs[i]).item()
            predicted_label = class_labels[predicted_label_index]
            predicted_probability = probs[i][predicted_label_index].item() * 100  # Convert to percentage

            print(f"True Label: {class_labels[true_label]}, Predicted Label: {predicted_label}, "
                  f"Probability: {predicted_probability:.2f}%")
