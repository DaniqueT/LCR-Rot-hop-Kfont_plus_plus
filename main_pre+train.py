import argparse
import os
from typing import Optional
import xml.etree.ElementTree as ElementTree

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from model import EmbeddingsLayer, LCRRotHopPlusPlus
from utils import train_validation_split


def clean_data(year: int, phase: str):
    """Clean a SemEval dataset by removing opinions with implicit targets. This function returns the cleaned dataset."""
    filename = f"ABSA{year % 2000}_Restaurants_{phase}.xml"

    input_path = f"data/raw/{filename}"
    output_path = f"data/processed/{filename}"

    if os.path.isfile(output_path):
        print(f"Found cleaned file at {output_path}")
        return ElementTree.parse(output_path)

    tree = ElementTree.parse(input_path)

    # remove implicit targets
    n_null_removed = 0
    for opinions in tree.findall(".//Opinions"):
        for opinion in opinions.findall('./Opinion[@target="NULL"]'):
            opinions.remove(opinion)
            n_null_removed += 1

    # calculate descriptive statistics for remaining opinions
    n = 0
    n_positive = 0
    n_negative = 0
    n_neutral = 0
    for opinion in tree.findall(".//Opinion"):
        n += 1

        if opinion.attrib['polarity'] == "positive":
            n_positive += 1
        elif opinion.attrib['polarity'] == "negative":
            n_negative += 1
        elif opinion.attrib['polarity'] == "neutral":
            n_neutral += 1

    if n == 0:
        print(f"\n{filename} does not contain any opinions")
    else:
        print(f"\n{filename}")
        print(f"  Removed {n_null_removed} opinions with target NULL")
        print(f"  Total number of opinions remaining: {n}")
        print(f"  Fraction positive: {100 * n_positive / n:.3f} %")
        print(f"  Fraction negative: {100 * n_negative / n:.3f} %")
        print(f"  Fraction neutral: {100 * n_neutral / n:.3f} %")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path)
    print(f"Stored cleaned dataset in {output_path}")

    return tree


class CustomDataset(Dataset):
    def __init__(self, data, embeddings_layer, device):
        self.data = data
        self.embeddings_layer = embeddings_layer
        self.device = device
        self.sentences = data.findall('.//sentence')

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        node = self.sentences[idx]
        sentence = node.find('./text').text
        opinions = []

        for opinion in node.findall('.//Opinion'):
            target_from = int(opinion.attrib['from'])
            target_to = int(opinion.attrib['to'])
            polarity = opinion.attrib['polarity']

            labels = {'negative': 0, 'neutral': 1, 'positive': 2}
            label = labels.get(polarity)
            embeddings, target_pos, hops = self.embeddings_layer.forward(sentence, target_from, target_to)
            left_size = target_pos[0]
            right_size = len(embeddings) - target_pos[1]
            left = embeddings[:left_size]
            target = embeddings[target_pos[0]:target_pos[1]]
            right = embeddings[-right_size:]
            opinions.append(((left, target, right), label, hops))

        return opinions


def train_model(model, criterion, optimizer, train_loader, val_loader, embeddings_layer, n_epochs, device):
    best_accuracy = 0
    best_state_dict = None

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")

        model.train()
        train_loss = 0.0
        train_n_correct = 0
        train_n = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", unit='batch'):
            for (left, target, right), label, hops in batch:
                optimizer.zero_grad()
                output = model(left, target, right, hops)
                loss = criterion(output, torch.tensor([label], device=device))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_n_correct += (output.argmax(1) == torch.tensor([label], device=device)).type(torch.int).sum().item()
                train_n += 1

        train_accuracy = train_n_correct / train_n
        print(f"Train Loss: {train_loss/train_n:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_n_correct = 0
        val_n = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch+1}/{n_epochs}", unit='batch'):
                for (left, target, right), label, hops in batch:
                    output = model(left, target, right, hops)
                    loss = criterion(output, torch.tensor([label], device=device))

                    val_loss += loss.item()
                    val_n_correct += (output.argmax(1) == torch.tensor([label], device=device)).type(torch.int).sum().item()
                    val_n += 1

        val_accuracy = val_n_correct / val_n
        print(f"Validation Loss: {val_loss/val_n:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state_dict = model.state_dict()

    return best_state_dict


def stringify_float(value: float):
    return str(value).replace('.', '-')


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2016, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--phase", default="Test", help="The phase of the dataset (Train or Test)")
    parser.add_argument("--hops", default=3, type=int, help="The number of hops to use in the rotatory attention mechanism")
    parser.add_argument("--val-split", default=0.2, type=float, help="Validation split fraction")
    args = parser.parse_args()

    year: int = args.year
    phase: str = args.phase
    lcr_hops: int = args.hops
    dropout_rate = 0.5

    learning_rate = 0.005
    momentum = 0.99
    weight_decay = 0.00001
    n_epochs = 5
    val_split = args.val_split

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_default_device(device)

    # Clean the data
    data = clean_data(year, phase)

    # Initialize the embeddings layer and the model
    embeddings_layer = EmbeddingsLayer(device = device)
    model = LCRRotHopPlusPlus(hops=lcr_hops, dropout_prob=dropout_rate).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Split dataset into training and validation
    dataset = CustomDataset(data, embeddings_layer, device)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: [item for sublist in x for item in sublist])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: [item for sublist in x for item in sublist])

    best_state_dict = train_model(model, criterion, optimizer, train_loader, val_loader, embeddings_layer, n_epochs,
                                  device)

    # Save the trained model
    models_dir = os.path.join("data", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{year}_LCR_hops{lcr_hops}_dropout{stringify_float(dropout_rate)}.pt")
    with open(model_path, "wb") as f:
        torch.save(best_state_dict, f)
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()