import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src import config


class KCData(Dataset):
    """
    ((questions, concepts, selectmasks -> attention_mask), response)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        questions_embeddings: torch.Tensor,
        concepts_embeddings: torch.Tensor,
    ):
        self.data = data
        self.question_embeddings = questions_embeddings
        self.concept_embeddings = concepts_embeddings

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        question_embeddings = self.question_embeddings[row["questions"]]  # (num_questions, emb_dim)
        concept_embeddings = self.concept_embeddings[row["concepts"]]  # (num_questions, emb_dim)
        attention_mask = torch.Tensor(row["selectmasks"])
        response = torch.LongTensor(row["responses"])

        return {
            "question_embeddings": question_embeddings,
            "concept_embeddings": concept_embeddings,
            "attention_mask": attention_mask,
            "response": response,
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    question_embeddings = torch.stack([item["question_embeddings"] for item in batch])
    concept_embeddings = torch.stack([item["concept_embeddings"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch]) == 1
    response = torch.stack([item["response"] for item in batch])
    response[response == -1] = 0

    return {
        "question_embeddings": question_embeddings,
        "concept_embeddings": concept_embeddings,
        "attention_mask": attention_mask,
        "response": response,
    }


def get_data_loaders(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    question_embeddings: torch.Tensor,
    concept_embeddings: torch.Tensor,
    trainer_config: config.TrainerConfig,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = KCData(train_data, question_embeddings, concept_embeddings)
    valid_dataset = KCData(valid_data, question_embeddings, concept_embeddings)

    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_config.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader
