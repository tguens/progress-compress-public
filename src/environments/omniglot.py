"""AI-Generated code.

Omniglot data module for reproducing the Progress & Compress (Schwarz et al., 2018)
supervised learning experiment.

Setup: 50 sequential tasks, each task = one alphabet.
Within each alphabet the model classifies which character (0..num_chars-1).
Background set (30 alphabets) + Evaluation set (20 alphabets) = 50 alphabets.
Each character has 20 handwritten samples; we split 15 train / 5 test per character.
"""

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMG_SIZE = 28
TRAIN_DRAWERS = 15
TEST_DRAWERS = 5
TOTAL_DRAWERS = 20


@dataclass
class AlphabetTask:
    """One continual-learning task = one Omniglot alphabet."""

    task_id: int
    alphabet_name: str
    num_classes: int  # number of characters in this alphabet
    train_dataset: Dataset
    test_dataset: Dataset

    def train_loader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """With shuffle=True (default), each new DataLoader gets a reshuffled minibatch order."""
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)

    def test_loader(self, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)


class AlphabetDataset(Dataset):
    """Wraps a subset of Omniglot samples for one alphabet with local class labels."""

    def __init__(self, images: list[torch.Tensor], labels: list[int], transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def _build_alphabet_index(dataset: datasets.Omniglot) -> dict[str, list[int]]:
    """Map alphabet name -> sorted list of character indices in dataset._characters."""
    alpha_to_char_indices: dict[str, list[int]] = defaultdict(list)
    for char_idx, char_path in enumerate(dataset._characters):
        alphabet_name = char_path.split("/")[0]
        alpha_to_char_indices[alphabet_name].append(char_idx)
    return dict(alpha_to_char_indices)


def _build_sample_index(dataset: datasets.Omniglot) -> dict[int, list[int]]:
    """Map global character index -> list of dataset sample indices."""
    char_to_samples: dict[int, list[int]] = defaultdict(list)
    for sample_idx in range(len(dataset)):
        _, char_label = dataset[sample_idx]
        char_to_samples[char_label].append(sample_idx)
    return dict(char_to_samples)


def load_tasks(
    data_root: str = "./data",
    seed: int = 0,
    data_augmentation: bool = False,
) -> list[AlphabetTask]:
    """
    Load all 50 Omniglot alphabet tasks.

    Returns a list of 50 AlphabetTask objects, each with train/test datasets.
    Task order: background alphabets (sorted) then evaluation alphabets (sorted),
    then shuffled with the given seed for reproducibility.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]
    )
    train_augment = (
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
        if data_augmentation
        else None
    )

    bg = datasets.Omniglot(
        root=data_root, background=True, download=True, transform=transform
    )
    ev = datasets.Omniglot(
        root=data_root, background=False, download=True, transform=transform
    )

    tasks: list[AlphabetTask] = []

    for split_idx, dataset in enumerate([bg, ev]):
        alpha_to_chars = _build_alphabet_index(dataset)
        char_to_samples = _build_sample_index(dataset)

        for alphabet_name in sorted(alpha_to_chars.keys()):
            char_indices = alpha_to_chars[alphabet_name]
            num_classes = len(char_indices)

            train_images, train_labels = [], []
            test_images, test_labels = [], []

            for local_label, global_char_idx in enumerate(sorted(char_indices)):
                sample_indices = char_to_samples[global_char_idx]
                assert len(sample_indices) == TOTAL_DRAWERS

                for drawer_idx, sample_idx in enumerate(sample_indices):
                    img, _ = dataset[sample_idx]
                    if drawer_idx < TRAIN_DRAWERS:
                        train_images.append(img)
                        train_labels.append(local_label)
                    else:
                        test_images.append(img)
                        test_labels.append(local_label)

            task = AlphabetTask(
                task_id=len(tasks),
                alphabet_name=alphabet_name,
                num_classes=num_classes,
                train_dataset=AlphabetDataset(
                    train_images, train_labels, transform=train_augment
                ),
                test_dataset=AlphabetDataset(test_images, test_labels),
            )
            tasks.append(task)

    # Shuffle task order with seed for reproducibility
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(tasks), generator=rng).tolist()
    tasks = [tasks[i] for i in perm]
    for new_id, task in enumerate(tasks):
        task.task_id = new_id

    return tasks


def max_classes(tasks: list[AlphabetTask]) -> int:
    """Max number of characters in any alphabet, useful for sizing the output head."""
    return max(t.num_classes for t in tasks)


# TODO: Do we need this?
if __name__ == "__main__":
    print("Loading 50 Omniglot alphabet tasks...")
    tasks = load_tasks()

    print(f"\n{'ID':>3}  {'Alphabet':<50} {'Classes':>7} {'Train':>6} {'Test':>5}")
    print("-" * 80)
    total_train, total_test = 0, 0
    for t in tasks:
        n_train = len(t.train_dataset)  # type: ignore
        n_test = len(t.test_dataset)  # type: ignore
        total_train += n_train
        total_test += n_test
        print(
            f"{t.task_id:>3}  {t.alphabet_name:<50} {t.num_classes:>7} "
            f"{n_train:>6} {n_test:>5}"
        )
    print("-" * 80)
    print(
        f"     {'TOTAL':<50} {max_classes(tasks):>7} {total_train:>6} {total_test:>5}"
    )
    print(f"\nMax classes in any alphabet: {max_classes(tasks)}")

    # Quick loader sanity check
    t0 = tasks[0]
    loader = t0.train_loader(batch_size=16)
    batch_img, batch_lbl = next(iter(loader))
    print(f"\nTask 0 ({t0.alphabet_name}) sample batch:")
    print(
        f"  images: {batch_img.shape}  labels: {batch_lbl.shape} unique: {batch_lbl.unique().tolist()}"
    )
