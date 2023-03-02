import math
from functools import cached_property, reduce, partial
from typing import Callable, Any, ClassVar, Optional
from unicodedata import normalize
from pathlib import Path
from dataclasses import dataclass, asdict
from textwrap import dedent
from datetime import date
import random
import string

from timeit import default_timer as timer

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)

import polars as pl
import duckdb


class CharacterTokenizer:
    def __init__(self, text: str):
        self.char_to_idx = {char: idx for idx, char in enumerate(text)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def tokenize(self, text: str):
        return [self.char_to_idx[char] for char in text]

    def untokenize(self, tokens: list[float]):
        return "".join([self.idx_to_char[token] for token in tokens])


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 64):
        super().__init__()

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
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


@dataclass
class DateParsingTransformerConfig:  # pylint: disable=too-many-instance-attributes
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    emb_size: int = 32
    nhead: int = 4
    dim_feedforward: int = 32
    vocab_size: Optional[int] = None
    src_vocab_size: Optional[int] = None
    tgt_vocab_size: Optional[int] = None
    dropout: Optional[float] = 0.1


class DateParsingTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        dim_feedforward: int,
        vocab_size: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dropout: float,
    ):
        super().__init__()
        # TODO(@axdg): Add a property that will allow us to retrieve a
        # "device" - so that we don't need to constantly pass it around.
        src_vocab_size = tgt_vocab_size or vocab_size
        tgt_vocab_size = tgt_vocab_size or vocab_size

        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(
            # TODO(@axdg): Rename this parameter to `d_model` in order to
            # be consistent with the original paper / the PyTorch implementation.
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )

        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )

        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    @cached_property
    @staticmethod
    def device():
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    @cached_property
    def size(self):
        return (
            sum(p.numel() for p in self.parameters()),
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

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
        # TODO(@axdg): Several things:
        #   - [ ] The "encodings" should actually be called `positional_embedding`s.
        #     The class name for these should be `PositionalEmbedding`.
        #   - [ ] We need to nicely handle keyboard interrupts, and add methods
        #     for actually testing these functions out.
        #   - [ ] We need to add a method for saving the model.
        #   - [x] ~We need to add a method for loading the model; done - this one
        #     is done at weight initialization.~
        #   - [ ] We need to add the tokenizer padding tokens as a method on this
        #     class.
        #   - [ ] We need to replace the custom `generate_square_subsequent_mask`
        #     with the one that exists as a static method on the `Transformer`
        #     class that PyTorch provides.
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

        memory = self.encoder(src_emb, src_mask, src_padding_mask)
        outs = self.decoder(
            tgt_emb, memory, tgt_mask, None, tgt_padding_mask, memory_key_padding_mask
        )

        return self.generator(outs)

    def initialize_weights(self, path: Path = None):
        # TODO(@axdg): This should also allow for initialization from a file.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device):
        # TODO(@axdg): This method appears to exist as a static method of
        # `torch.nn.Transformer` - so we should probably use that instead.
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )

        return mask

    @staticmethod
    def create_mask(src: Tensor, tgt: Tensor, pad_idx: int, device):
        # TODO(@axdg): This is not aptly named - and it won't suite our
        # purpose in a context where we're wanting to use multiple
        # decoders. We should split this into two functions for the src and
        # target masks.
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = DateParsingTransformer.generate_square_subsequent_mask(
            tgt_seq_len, device
        )
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
class AdamConfig:
    lr: float = 0.0001
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9


@dataclass
class RMSpropConfig:
    lr: float = 0.0001
    alpha: float = 0.99
    eps: float = 1e-08
    weight_decay: float = 0
    momentum: float = 0
    centered: bool = False


@dataclass
class RandomDateDataset(Dataset):
    DATE_FORMATS: ClassVar[list[str]] = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%a %d/%m/%Y",
        "%a %d-%m-%Y",
        "%a %d.%m.%Y",
        "%A %d/%m/%Y",
        "%A %d-%m-%Y",
        "%A %d.%m.%Y",
        "%A, %d %B %Y",
        "%A, %B %d %Y",
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

            df = duckdb.sql(query=SCRIPT).pl()  # pylint: disable=c-extension-no-member
            df = df.select(
                [
                    pl.col("d")
                    .apply(lambda d: d.strftime(random.choice(self.DATE_FORMATS)))
                    .alias("x"),
                    pl.col("d").dt.strftime(fmt="%Y-%m-%d").alias("y"),
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
    text = text.upper()
    text = text.strip()
    return text


def create_tensor(v: list[int], bos_idx: str, eos_idx: str) -> torch.Tensor:
    """
    Creates a tensor from a list of integers, and adds the BOS and EOS
    tokens at the appropriate positions.
    """
    return torch.cat(
        (torch.tensor([bos_idx]), torch.tensor(v), torch.tensor([eos_idx]))
    )


DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
STATE_DICT_PATH = DATA_DIR / "dpt.pt"
STATE_DICT_PATH_NEXT = DATA_DIR / "dpt-next.pt"


def main(num_epochs: int = 18, batch_size: int = 128):
    # TODO(@axdg): This whole thing needs to be broken up into a bunch
    # of different functions.
    torch.manual_seed(42)

    DEVICE = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    CHARACTERS = "!:@|" + string.ascii_uppercase + string.digits + " -.,/\\"

    # TODO(@axdg): We could probably make this a parameter on the tokenizer
    # or the model instance.
    PAD_IDX = CHARACTERS.index("!")
    BOS_IDX = CHARACTERS.index(":")
    EOS_IDX = CHARACTERS.index("@")

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
        vocab_size=tokenizer.vocab_size,
    )

    transformer = DateParsingTransformer(**asdict(config))

    # TODO(@axdg): Replace this with something better (a method on the original
    # class). Or just something better like `transformer.init_weights()`.
    transformer.load_state_dict(torch.load(STATE_DICT_PATH_NEXT))

    # TODO(@axdg): This was originally train with:
    # ```
    # optimizer = torch.optim.RMSprop(
    #     transformer.parameters(),
    #     **asdict(RMSpropConfig()),
    # )
    # ```
    # Consider switching back to that (or experiment more).

    optimizer = torch.optim.Adam(transformer.parameters(), **asdict(AdamConfig()))

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    train_dataset = RandomDateDataset(size=50_000)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )

    val_dataset = RandomDateDataset(size=3000)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn
    )

    transformer = transformer.to(DEVICE)
    criterion = criterion.to(DEVICE)

    def greedy_decode(model: nn.Module, src, src_mask, max_len, bos_idx):
        src = src.to(DEVICE)
        src_mask = src_mask.to(DEVICE)

        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(bos_idx).type(torch.long).to(DEVICE)

        for _ in range(max_len - 1):
            memory = memory.to(DEVICE)
            tgt_mask = (
                DateParsingTransformer.generate_square_subsequent_mask(
                    ys.size(0), DEVICE
                ).type(torch.bool)
            ).to(DEVICE)

            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)

            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            next_word = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0
            )
            if next_word == EOS_IDX:
                break
        return ys

    def translate(model: nn.Module, src_sentence: str):
        model.eval()
        src = transform_text_to_sequence(src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = greedy_decode(
            model, src, src_mask, max_len=num_tokens + 5, bos_idx=BOS_IDX
        ).flatten()

        tgt_chars = tokenizer.untokenize(list(tgt_tokens.cpu().numpy()))
        return "".join(tgt_chars).replace(":", "").replace("@", "")

    TEST_DATES = [date.today().strftime(f) for f in RandomDateDataset.DATE_FORMATS]

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
            dataloader=val_dataloader,
            pad_idx=PAD_IDX,
            device=DEVICE,
        )

        print(
            (
                f"EPOCH: {epoch + 1}, LOSS: {train_loss:.3f}, VAL LOSS: {val_loss:.3f}, "
                f"({(end_time - start_time):.3f}s)"
            )
        )

        d: str
        for d in TEST_DATES:
            s = timer()
            p = translate(transformer, d).strip()
            e = timer()

            c = p == date.today().strftime("%Y-%m-%d")

            print(
                d.ljust(36, " "),
                p.ljust(15, " "),
                " Y " if c else " N ",
                f"{math.floor(((e - s) * 1000))}ms".ljust(10, " "),
            )

        torch.save(transformer.state_dict(), STATE_DICT_PATH_NEXT)


if __name__ == "__main__":
    main(num_epochs=20, batch_size=128)
