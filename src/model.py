import torch
import torch.nn as nn

from src import config


class ProjectInput(nn.Module):
    def __init__(self, model_config: config.KCModelConfig):
        super().__init__()
        self.question_projector = nn.Linear(model_config.embedding_dim, model_config.hidden_size)
        self.concept_projector = nn.Linear(model_config.embedding_dim, model_config.hidden_size)
        self.responses = nn.Embedding(2, model_config.hidden_size)
        self.start_token = nn.Embedding(1, model_config.hidden_size)

        self.dropout = nn.Dropout(0.2)
        self.activation = nn.GELU()

        self.layer_norm = nn.LayerNorm(3 * model_config.hidden_size)

        self.concatenated_projector = nn.Linear(
            3 * model_config.hidden_size, model_config.hidden_size
        )
        self.concatenated_layer_norm = nn.LayerNorm(model_config.hidden_size)

        self.projection1 = nn.Linear(model_config.hidden_size, model_config.hidden_size)

    def forward(
        self, questions: torch.Tensor, concepts: torch.Tensor, responses: torch.Tensor
    ) -> torch.Tensor:
        questions = self.question_projector(questions)  # (batch_size, seq_len, hidden_size)
        concepts = self.concept_projector(concepts)  # (batch_size, seq_len, hidden_size)
        responses = self.responses(responses)  # (batch_size, seq_len, hidden_size)
        if responses.shape[1] == questions.shape[1]:
            responses = responses[:, :-1, :]  # drop the last response

        start_token_idx = torch.zeros(1, dtype=torch.long, device=questions.device)
        start_token_embedding: torch.Tensor = self.start_token(start_token_idx)
        responses_with_start = torch.cat(
            [
                start_token_embedding.expand(len(questions), 1, -1),
                responses,
            ],
            dim=1,
        )  # (batch_size, seq_len + 1, hidden_size)
        concatenated = torch.cat([questions, concepts, responses_with_start], dim=-1)

        x = self.concatenated_projector(self.dropout(self.layer_norm(concatenated)))
        x = self.concatenated_layer_norm(self.activation(x))
        x = self.projection1(x)

        return x


class KCLSTMModel(nn.Module):
    def __init__(self, model_config: config.KCModelConfig):
        super().__init__()
        self.num_layers = model_config.num_layers
        self.hidden_size = model_config.hidden_size
        self.input_projector = ProjectInput(model_config)
        self.lstm = nn.LSTM(
            model_config.hidden_size,
            model_config.hidden_size,
            model_config.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(model_config.hidden_size, 1)

    def forward(
        self, x: dict[str, torch.Tensor], return_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.input_projector(x["question_embeddings"], x["concept_embeddings"], x["response"])
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)
        if return_state:
            return out, (hn, cn)
        else:
            return out
