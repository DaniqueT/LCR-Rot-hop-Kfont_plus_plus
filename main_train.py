import argparse
import os
from typing import Optional

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDataset, train_validation_split
from pytorchtools import EarlyStopping

import numpy as np
import gensim
from rdflib import Graph
import random


def stringify_float(value: float):
    return str(value).replace('.', '-')

def identity_collate(batch):
    return batch


def main():
    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--year",
        default=2016,
        type=int,
        help="The year of the dataset (2015 or 2016)"
    )

    parser.add_argument(
        "--hops",
        default=3,
        type=int,
        help="The number of hops to use in the rotatory attention mechanism"
    )

    args = parser.parse_args()

    year: int = args.year
    lcr_hops: int = args.hops

    dropout_rate = 0.5
    learning_rate = 0.02
    momentum = 0.9
    weight_decay = 0.025
    n_epochs = 40
    batch_size = 32

    patience = 10


    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )

    train_dataset = EmbeddingsDataset(year=year, device=device, phase="Train")
    print(f"Loaded {train_dataset} with {len(train_dataset)} obs in total")

    train_idx, validation_idx = train_validation_split(train_dataset)

    training_subset = Subset(train_dataset, train_idx)
    validation_subset = Subset(train_dataset, validation_idx)
    print(f"Using {train_dataset} with {len(training_subset)} obs for training")
    print(f"Using {train_dataset} with {len(validation_subset)} obs for validation")

    training_loader = DataLoader(
        training_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=identity_collate
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        collate_fn=identity_collate
    )

    word2vec = gensim.models.Word2Vec.load("data/myontology2016/output")
    ontology = Graph().parse("data/raw/ontology.owl")


    model = LCRRotHopPlusPlus(
        hops=lcr_hops,
        dropout_prob=dropout_rate,
        word2vec=word2vec,
        ontology=ontology
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
         model.parameters(),
         lr=learning_rate,
         momentum=momentum,
         weight_decay=weight_decay
     )

    best_accuracy: Optional[float] = None
    best_state_dict: Optional[dict] = None

    epochs_progress = tqdm(range(n_epochs), unit='epoch')

    models_dir = os.path.join("data", "modelsLayers")
    os.makedirs(models_dir, exist_ok=True)
    name = "trainhop2.pt"

    model_path = os.path.join(
        models_dir,
        f"{year}_drop{stringify_float(dropout_rate)}_lr{stringify_float(learning_rate)}_wd{stringify_float(weight_decay)}_batch{stringify_float(batch_size)}_acc{stringify_float(best_accuracy)}_{name}"
    )

    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=model_path
    )

    try:
        for epoch in epochs_progress:

            model.train()

            train_loss = 0.0
            train_n_correct = 0
            train_n = 0

  
            for batch in tqdm(training_loader, unit='batch', leave=False):

                knowledge_layers = range(9, 12)
                outputs = []
                labels = []

                for (sentence, start, end), label, _ in batch:
                    outputs.append(model(sentence, start, end, knowledge_layers))
                    labels.append(label)

                batch_outputs = torch.stack(outputs)
                batch_labels = torch.stack(labels).to(device)

                loss = criterion(batch_outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch)

                train_n_correct += (
                    (batch_outputs.argmax(1) == batch_labels)
                    .sum()
                    .item()
                )

                train_n += len(batch)

            avg_train_loss = train_loss / train_n
            train_accuracy = train_n_correct / train_n

            epochs_progress.set_description(
                f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                f"Train Acc={train_accuracy:.4f}"
            )

            model.eval()

            val_loss = 0.0
            val_n_correct = 0
            val_n = 0

            with torch.inference_mode():
                for batch in tqdm(validation_loader, unit='batch', leave=False):

                    outputs = torch.stack(
                    [
                        model(sentence, start, end, knowledge_layers = range(-2, -1))
                        for (sentence, start, end), _, hops in batch
                    ],
                    dim=0
                )

                    labels = torch.tensor(
                        [label.item() for _, label, _ in batch],
                        device=device
                    )

                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * len(batch)

                    val_n_correct += (
                        outputs.argmax(1) == labels
                    ).sum().item()

                    val_n += len(batch)

            avg_val_loss = val_loss / val_n
            val_accuracy = val_n_correct / val_n

            print(
                f"Validation Loss={avg_val_loss:.4f}, "
                f"Val Acc={val_accuracy:.4f}"
                f" van_n_corrct= {val_n_correct}", f" val_n= {val_n}"
            )

            # Early stopping
            early_stopping(avg_val_loss, model)

            # Save best model by accuracy
            if best_accuracy is None or val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_state_dict = model.state_dict()

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    except KeyboardInterrupt:
        print("Interrupted training procedure, saving best model...")

    if best_state_dict is not None:

        models_dir = os.path.join("data", "modelsLayers")
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(
            models_dir,
             f"{year}_drop{stringify_float(dropout_rate)}_lr{stringify_float(learning_rate)}_wd{stringify_float(weight_decay)}_batch{stringify_float(batch_size)}_acc{stringify_float(round(best_accuracy, 4))}_{name}"
    )

        with open(model_path, "wb") as f:
            torch.save(best_state_dict, f)

        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()