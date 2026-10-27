import io
import string
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app import config, extraction


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DONUT_PO_DATA_DIR", str(tmp_path / "data"))
    extraction._cache.update(release_id=None, extractor=None)
    return tmp_path / "data"


def po_image(lines: list[str], size=(480, 640)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((20, 20 + i * 24), line, fill="black")
    return img


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def pdf_bytes(pages: list[list[str]]) -> bytes:
    import fitz

    doc = fitz.open()
    for lines in pages:
        page = doc.new_page(width=420, height=595)
        for i, line in enumerate(lines):
            page.insert_text((40, 60 + i * 22), line, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def sample_values(n: int, layout: str) -> dict:
    return {
        "po_number": f"PO-{n:03d}",
        "po_date": "01/02/2026",
        "customer_name": layout,
        "items": [{"description": "Gula pasir", "quantity": str(n), "unit_price": "12.500"}],
        "total_amount": f"{12500 * n:,}",
    }


@pytest.fixture(scope="session")
def tiny_donut(tmp_path_factory) -> Path:
    """A very small, randomly initialised Donut model (no download needed)."""
    pytest.importorskip("torch")
    import sentencepiece as spm
    from transformers import (
        DonutImageProcessor, DonutProcessor, DonutSwinConfig, DonutSwinModel, MBartConfig, MBartForCausalLM,
        VisionEncoderDecoderModel, XLMRobertaTokenizer,
    )

    d = tmp_path_factory.mktemp("tiny_donut")
    chars = string.ascii_letters + string.digits + ".,-/:() "
    corpus = d / "corpus.txt"
    corpus.write_text("\n".join([chars] * 5 + ["PO Gula pasir 12.500 01/02/2026 Toko Maju PT Sinar"] * 50))
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(d / "spm"), vocab_size=120, character_coverage=1.0, model_type="unigram",
        hard_vocab_limit=False,
    )
    tok = XLMRobertaTokenizer(vocab_file=str(d / "spm.model"))
    proc = DonutProcessor(
        image_processor=DonutImageProcessor(size={"height": 64, "width": 48}, do_align_long_axis=False), tokenizer=tok
    )
    enc = DonutSwinConfig(image_size=[64, 48], patch_size=4, embed_dim=16, depths=[1, 1], num_heads=[1, 2], window_size=4)
    dec = MBartConfig(
        vocab_size=len(tok), d_model=32, decoder_layers=1, decoder_attention_heads=2, decoder_ffn_dim=64,
        encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=64, max_position_embeddings=256,
        is_decoder=True, add_cross_attention=True, scale_embedding=True, add_final_layer_norm=True,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
    )
    model = VisionEncoderDecoderModel(encoder=DonutSwinModel(enc), decoder=MBartForCausalLM(dec))
    model.config.pad_token_id = tok.pad_token_id
    model.config.decoder_start_token_id = tok.bos_token_id
    out = d / "model"
    model.save_pretrained(out)
    proc.save_pretrained(out)
    return out


@pytest.fixture
def tiny_settings(tiny_donut, monkeypatch):
    monkeypatch.setattr(config, "BASE_MODEL", str(tiny_donut))
    monkeypatch.setattr(config, "IMAGE_HEIGHT", 64)
    monkeypatch.setattr(config, "IMAGE_WIDTH", 48)
    monkeypatch.setattr(config, "MAX_TARGET_LENGTH", 128)
    return tiny_donut
