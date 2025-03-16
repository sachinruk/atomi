import dataclasses
import pathlib

import pydantic


@dataclasses.dataclass
class DataConfig:
    columns: list[str] = dataclasses.field(
        default_factory=lambda: [
            "questions",
            "concepts",
            "responses",
            "timestamps",
            "selectmasks",
            "is_repeat",
        ]
    )
    train_data_path: pathlib.Path = pathlib.Path("data/XES3G5M/kc_level/train_valid_sequences.csv")
    test_data_path: pathlib.Path = pathlib.Path("data/XES3G5M/kc_level/test.csv")
    questions_embeddings_path: pathlib.Path = pathlib.Path(
        "data/XES3G5M/metadata/embeddings/qid2content_emb.json"
    )
    concepts_embeddings_path: pathlib.Path = pathlib.Path(
        "data/XES3G5M/metadata/embeddings/cid2content_emb.json"
    )


@dataclasses.dataclass
class WandbConfig:
    WANDB_LOG_PATH: pathlib.Path = pathlib.Path("/tmp/wandb_logs")
    WANDB_LOG_PATH.mkdir(parents=True, exist_ok=True)
    WANDB_ENTITY: str = "sachinruk"


class TrainerConfig(pydantic.BaseModel):
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 10

    model_name: str = "kc_model"
    project_name: str = "kc_project"
    is_local: bool = True

    class Config:
        extra = "forbid"
        protected_namespaces = ()
