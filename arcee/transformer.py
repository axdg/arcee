import math
from typing import Iterable, List

from timeit import default_timer as timer

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Transformer

from dataclasses import dataclass, asdict
from textwrap import dedent
from datetime import date
import random
import string

from typing import ClassVar

import pyarrow
import polars as pl

import duckdb
from duckdb import DuckDBPyConnection

import torch
from torch.utils.data import DataLoader, Dataset

from functools import reduce, partial
from typing import Callable, Any
from torch.nn.utils.rnn import pad_sequence
from unicodedata import normalize


class CharacterTokenizer:
    def __init__(self, text: str):
        self.char_to_idx = {char: idx for idx, char in enumerate(text)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def __len__(self):
        return len(self.char_to_idx)

    @property
    def vocab_size(self) -> int:
        return len(self)

    def tokenize(self, text: str):
        return [self.char_to_idx[char] for char in text]

    def untokenize(self, tokens: list[float]):
        return "".join([self.idx_to_char[token] for token in tokens])


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class DateParsingTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(DateParsingTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )

        return mask

    @staticmethod
    def create_mask(src: Tensor, tgt: Tensor, pad_idx: int, device):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = DateParsingTransformer.generate_square_subsequent_mask(tgt_seq_len, device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(
            torch.bool
        )

        src_padding_mask = (src == pad_idx).transpose(0, 1)
        tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    dataloader: DataLoader,
    pad_idx: int,
    device,
):
    model.train()

    count = 0
    losses = 0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        (
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
        ) = DateParsingTransformer.create_mask(src, tgt_input, pad_idx, device)

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        count = count + 1
        if count % 10 == 0:
            print("RUNNING ...")

    return losses / len(list(dataloader))


def evaluate(
    model: nn.Module,
    criterion: Callable,
    dataloader: DataLoader,
    pad_idx: int,
    device,
):
    model.eval()
    losses = 0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        (
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
        ) = DateParsingTransformer.create_mask(src, tgt_input, pad_idx, device)

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(dataloader))


@dataclass
class DateParsingTransformerConfig:
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    emb_size: int = 192
    nhead: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1
    src_vocab_size: int = 10
    tgt_vocab_size: int = 10


@dataclass
class AdamConfig:
    lr: float = 0.0001
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9


@dataclass
class RandomDateDataset(Dataset):
    DATE_FORMATS: ClassVar[list[str]] = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%m.%d.%Y",
        "%A, %d %B %Y",
        "%A, %B %d %Y",
        "%a, %d %d %Y",
        "%a, %B %d %Y",
        "%A %d %B %Y",
        "%A %B %d %Y",
        "%a %B %d %Y",
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%Y.%m.%d",
        "%d %B %Y",
        "%B %d, %Y",
        "%d %b %Y",
        "%Y-%b-%d",
        "%b %d, %Y",
        "%b %d %Y",
        "%d-%b-%Y",
        "%b-%d-%Y",
        "%Y/%b/%d",
        "%Y.%b.%d",
        "%B %d %Y",
        "%b %d %Y",
    ]

    _df: pl.DataFrame = None

    size: int = 1000
    start_date: date = date(1800, 1, 1)
    end_date: date = date(2100, 12, 31)

    @property
    def df(self) -> pl.DataFrame:
        if self._df is None:
            SCRIPT = dedent(
                """\
                WITH record AS (
                    SELECT
                        RANDOM() AS r,
                        DATE_DIFF('day', DATE '{start_date}', DATE '{end_date}') AS n,
                    FROM
                        GENERATE_SERIES(1, {n})
                )

                SELECT
                    DATE '1800-01-01' + CAST(FLOOR(r * n) AS INTEGER) AS d,
                FROM
                    record
            """
            ).format(
                n=self.size,
                start_date=str(self.start_date),
                end_date=str(self.end_date),
            )

            df = duckdb.sql(query=SCRIPT).pl()
            df = df.select(
                [
                    pl.col("d")
                    .apply(lambda d: d.strftime(random.choice(self.DATE_FORMATS)))
                    .alias("x"),
                    pl.col("d").dt.strftime(fmt="%Y-%m-%d|%w").alias("y"),
                ]
            )

            self._df = df

        return self._df

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.df.row(idx)


def pipe(*args: list[Callable[[Any], Any]]):
    return reduce(lambda f, g: lambda x: f(g(x)), reversed(args))


def preprocess_text(text: str) -> str:
    """
    A function that is used to preprocess the text that's fed to the model.

    The function operates on single examples (of dates). It's required for
    input data - since we don't need to support all casings (so we
    convert everything to uppercase) etc... it's not needed for target data
    during training - although it's incidentally used is will be a no-op on
    the text that we've generated as targets.

    Args:
        text: The text to preprocess.

    # TODO(@axdg): Consider adding some extra operations to this function -
    # while dates are a contrived example (and so don't need anything extra)
    # it would be worthwhile to add examples of stripping useless
    # characters, normalising ambiguous characters (e.g. 0 vs O) and
    # duplicated spaces etc.
    """
    text = normalize("NFC", text)
    text = text.upper().strip()
    return text


def create_tensor(v: list[int], bos_idx: str, eos_idx: str) -> torch.Tensor:
    """
    Creates a tensor from a list of integers, and adds the BOS and EOS
    tokens at the appropriate positions.
    """
    return torch.cat(
        (torch.tensor([bos_idx]), torch.tensor(v), torch.tensor([eos_idx]))
    )


def main(num_epochs: int = 18, batch_size: int = 128):
    torch.manual_seed(42)

    DEVICE = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    CHARACTERS = "!:@|" + string.ascii_uppercase + string.digits + " -.,/\\"

    # TODO(@axdg): We could probably make the parameter on the tokenizer.
    PAD_IDX = CHARACTERS.index("!")
    BOS_IDX = CHARACTERS.index(":")
    EOS_IDX = CHARACTERS.index("@")
    SEP_IDX = CHARACTERS.index("|")

    tokenizer = CharacterTokenizer(CHARACTERS)

    _create_tensor = partial(create_tensor, bos_idx=BOS_IDX, eos_idx=EOS_IDX)
    _pad_sequence = partial(pad_sequence, padding_value=PAD_IDX)

    transform_text_to_sequence = pipe(
        preprocess_text, tokenizer.tokenize, _create_tensor
    )

    def collate_fn(batch: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = [], []

        for src, tgt in batch:
            x.append(transform_text_to_sequence(src))
            y.append(transform_text_to_sequence(tgt))

        return _pad_sequence(x), _pad_sequence(y)

    config = DateParsingTransformerConfig(
        src_vocab_size=len(tokenizer), tgt_vocab_size=len(tokenizer)
    )

    transformer = DateParsingTransformer(**asdict(config))
    transformer.init_weights()

    optimizer = torch.optim.Adam(transformer.parameters(), **asdict(AdamConfig()))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    train_dataset = RandomDateDataset(size=20_000)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )

    val_dataset = RandomDateDataset(size=1000)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn
    )

    transformer = transformer.to(DEVICE)
    criterion = criterion.to(DEVICE)

    for epoch in range(num_epochs):
        start_time = timer()
        train_loss = train(
            model=transformer,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_dataloader,
            pad_idx=PAD_IDX,
            device=DEVICE,
        )

        end_time = timer()

        val_loss = evaluate(
            model=transformer,
            criterion=criterion,
            dataloader=train_dataloader,
            pad_idx=PAD_IDX,
            device=DEVICE,
        )

        print(
            (
                f"EPOCH: {epoch + 1}, LOSS: {train_loss:.3f}, VAL LOSS: {val_loss:.3f}, "
                f"({(end_time - start_time):.3f}s)"
            )
        )


if __name__ == main():
    main(num_epochs=3, batch_size=256)
