"""Donut (OCR-free Document Understanding Transformer) wrapper.

Donut reads the page image directly — no separate OCR step — and generates a token sequence such as
  <s_po><s_po_number>PO-001</s_po_number><s_items><s_description>Gula</s_description>...</s_items>
which is converted back into JSON. Field tokens come from the PO Data Field Definition Sheet.
"""
from __future__ import annotations

import json
import math
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image

from . import config
from .field_definition import FieldDefinition

MODEL_META = "po_model_meta.json"


def _torch():
    import torch

    return torch


def pick_device() -> str:
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainItem:
    image: Path
    values: dict


def build_for_training(base_model: str, fd: FieldDefinition, image_size: tuple[int, int], max_length: int):
    """Load the pre-trained Donut checkpoint and adapt it to the PO field tokens."""
    from transformers import DonutProcessor, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

    cfg = VisionEncoderDecoderConfig.from_pretrained(base_model)
    cfg.encoder.image_size = list(image_size)  # (height, width)
    cfg.decoder.max_length = max_length
    processor = DonutProcessor.from_pretrained(base_model)
    model = VisionEncoderDecoderModel.from_pretrained(base_model, config=cfg)

    tok = processor.tokenizer
    new_tokens = [t for t in fd.special_tokens() if t not in tok.get_vocab()]
    if new_tokens:
        tok.add_special_tokens({"additional_special_tokens": new_tokens}, replace_additional_special_tokens=False)
        model.decoder.resize_token_embeddings(len(tok))

    processor.image_processor.size = {"height": image_size[0], "width": image_size[1]}
    processor.image_processor.do_align_long_axis = False

    model.config.pad_token_id = tok.pad_token_id
    model.config.decoder_start_token_id = tok.convert_tokens_to_ids(fd.task_token)
    model.config.eos_token_id = tok.eos_token_id
    return processor, model


class _Dataset:
    def __init__(self, items: list[TrainItem], processor, fd: FieldDefinition, max_length: int):
        self.items, self.processor, self.fd, self.max_length = items, processor, fd, max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        with Image.open(item.image) as img:
            pixel_values = self.processor(img.convert("RGB"), return_tensors="pt").pixel_values[0]
        target = self.fd.to_token_sequence(item.values) + self.processor.tokenizer.eos_token
        ids = self.processor.tokenizer(
            target, add_special_tokens=False, max_length=self.max_length, truncation=True, return_tensors="pt"
        ).input_ids[0]
        return pixel_values, ids


def _collate(batch, pad_id: int):
    torch = _torch()
    pixel_values = torch.stack([b[0] for b in batch])
    longest = max(len(b[1]) for b in batch)
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)  # -100 = ignored by the loss
    for i, (_, ids) in enumerate(batch):
        labels[i, : len(ids)] = ids
    return pixel_values, labels


def train(
    items: list[TrainItem],
    fd: FieldDefinition,
    out_dir: Path,
    *,
    base_model: str | None = None,
    epochs: int = 30,
    learning_rate: float = 3e-5,
    batch_size: int = 1,
    grad_accum: int = 1,
    image_size: tuple[int, int] | None = None,
    max_length: int | None = None,
    seed: int = 42,
    log: Callable[[str], None] = print,
    progress: Callable[[float], None] | None = None,
    stop_event: threading.Event | None = None,
) -> Path:
    """Fine-tune Donut on the training set and save the model to out_dir."""
    torch = _torch()
    from torch.utils.data import DataLoader

    random.seed(seed)
    torch.manual_seed(seed)
    base_model = base_model or config.BASE_MODEL
    image_size = tuple(image_size or (config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    max_length = max_length or config.MAX_TARGET_LENGTH

    log(f"Loading base model {base_model}")
    processor, model = build_for_training(base_model, fd, image_size, max_length)
    device = pick_device()
    model.to(device)
    log(f"Device: {device}; training pages: {len(items)}; epochs: {epochs}")

    ds = _Dataset(items, processor, fd, max_length)
    pad_id = processor.tokenizer.pad_token_id
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: _collate(b, pad_id))
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = max(1, epochs * math.ceil(len(loader) / grad_accum))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: max(0.0, 1 - s / total_steps))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    model.train()
    for epoch in range(1, epochs + 1):
        running, n = 0.0, 0
        for i, (pixel_values, labels) in enumerate(loader, 1):
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Training cancelled")
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = model(pixel_values=pixel_values, labels=labels).loss / grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
            if i % grad_accum == 0 or i == len(loader):
                if scaler:
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad()
                sched.step()
            running += loss.item() * grad_accum
            n += 1
        log(f"epoch {epoch}/{epochs} loss {running / max(n, 1):.4f}")
        if progress:
            progress(epoch / epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    (out_dir / MODEL_META).write_text(
        json.dumps(
            {
                "task_token": fd.task_token,
                "field_definition_hash": fd.hash,
                "image_size": list(image_size),
                "max_length": max_length,
                "base_model": base_model,
            },
            indent=2,
        )
    )
    log(f"Model saved to {out_dir}")
    return out_dir


class DonutExtractor:
    """Loads a trained PO model and reads page images into raw JSON."""

    def __init__(self, model_dir: Path, device: str | None = None):
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / MODEL_META).read_text())
        self.processor = DonutProcessor.from_pretrained(self.model_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_dir)
        self.device = device or pick_device()
        self.model.to(self.device).eval()
        self.lock = threading.Lock()

    def read_page(self, image: Image.Image | Path) -> tuple[dict, str]:
        """Return (parsed JSON, raw token sequence) for one page image."""
        torch = _torch()
        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                image = im.convert("RGB")
        tok = self.processor.tokenizer
        pixel_values = self.processor(image.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)
        prompt = tok(self.meta["task_token"], add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        max_len = min(self.meta["max_length"], self.model.decoder.config.max_position_embeddings)
        with self.lock, torch.no_grad():
            out = self.model.generate(
                pixel_values,
                decoder_input_ids=prompt,
                max_length=max_len,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[tok.unk_token_id]],
            )
        seq = self.processor.batch_decode(out)[0]
        seq = seq.replace(tok.eos_token, "").replace(tok.pad_token, "")
        seq = seq.replace(self.meta["task_token"], "", 1).strip()
        try:
            parsed = self.processor.token2json(seq)
        except Exception:  # noqa: BLE001 - malformed sequence from an under-trained model
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        return parsed, seq

    def read_pages(self, images: Iterable[Path]) -> list[tuple[dict, str]]:
        return [self.read_page(p) for p in images]


def merge_pages(fd: FieldDefinition, pages: list[dict]) -> dict:
    """Combine per-page results of a multi-page PO: first non-empty header value wins, line items are appended."""
    merged = fd.empty_record()
    for page in pages:
        for f in fd.fields:
            if f.is_list:
                merged[f.name].extend(page.get(f.name) or [])
            elif merged[f.name] in (None, "") and page.get(f.name) not in (None, ""):
                merged[f.name] = page[f.name]
    return merged
