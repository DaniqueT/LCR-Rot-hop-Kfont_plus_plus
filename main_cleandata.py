import argparse
import os
from typing import Optional
import xml.etree.ElementTree as ElementTree

import torch
from rdflib import Graph
from tqdm import tqdm

from model import EmbeddingsLayer
from utils import download_from_url, EmbeddingsDataset


def main(year: int, phase: str):
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

if __name__ == "__main__":
    main(2016, 'Test')
