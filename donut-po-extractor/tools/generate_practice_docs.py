"""Generate fictional customer purchase orders for practice (training + extraction test).

    python tools/generate_practice_docs.py [output_dir]      # default: practice_docs/

Creates 3 customer layouts with different designs, languages, date and number formats:
  * PT Sumber Makmur Sejahtera  — Indonesian, bordered table, dd/mm/yyyy, 1.234.500     (PDF)
  * CV Berkah Jaya Abadi        — English, dd Mon yyyy, 1,234,500.00, some 2-page POs  (PDF)
  * Koperasi Karyawan Sentosa   — compact, yyyy-mm-dd, Rp 1.234.500, scanned look      (JPG)
plus one layout that is NOT in the training set (Toko Serba Ada Mulia) to see the BUC 3 warning.

Each folder has answer_key.json with the correct field values per page, written exactly as printed.
All companies, people and numbers are made up.
"""
from __future__ import annotations

import io
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageFilter

SUPPLIER = ("PT Warung Distribusi Nusantara", "Jl. Gudang Raya No. 8, Bekasi")

CATALOG = [
    ("BRS-005", "Beras Premium 5 kg", "SAK", 72500),
    ("GLP-001", "Gula Pasir 1 kg", "PCS", 16500),
    ("MYG-002", "Minyak Goreng 2 L", "PCS", 34000),
    ("TPG-001", "Tepung Terigu 1 kg", "PCS", 12500),
    ("KOP-200", "Kopi Bubuk 200 g", "PCS", 18750),
    ("TEH-025", "Teh Celup isi 25", "BOX", 7800),
    ("MIE-040", "Mie Instan Goreng (karton 40)", "KRT", 112000),
    ("SUS-001", "Susu UHT 1 L", "PCS", 19900),
    ("SBN-090", "Sabun Mandi 90 g", "PCS", 4200),
    ("DTG-800", "Deterjen Bubuk 800 g", "PCS", 23500),
    ("AIR-600", "Air Mineral 600 ml (dus 24)", "DUS", 48000),
    ("KCP-520", "Kecap Manis 520 ml", "BTL", 21000),
    ("SAO-340", "Saus Sambal 340 ml", "BTL", 13500),
    ("GRM-500", "Garam Beryodium 500 g", "PCS", 5500),
    ("BSK-300", "Biskuit Kelapa 300 g", "PCS", 11200),
    ("PST-190", "Pasta Gigi 190 g", "PCS", 14800),
    ("TSU-250", "Tisu Wajah 250 lembar", "PCS", 16000),
    ("SRP-400", "Sirup Rasa Jeruk 460 ml", "BTL", 26500),
]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

LAYOUTS = {
    "PT Sumber Makmur Sejahtera": {
        "slug": "sumber_makmur", "style": "A", "prefix": "SMS/PO/",
        "address": "Jl. Merdeka No. 45, Bandung 40111", "delivery": "Gudang SMS, Jl. Soekarno-Hatta No. 210, Bandung",
        "terms": ["30 hari", "45 hari", "COD"],
    },
    "CV Berkah Jaya Abadi": {
        "slug": "berkah_jaya", "style": "B", "prefix": "BJA-",
        "address": "Jl. Pahlawan No. 17, Semarang 50241", "delivery": "Toko BJA Cabang Ungaran, Jl. Diponegoro No. 5, Ungaran",
        "terms": ["Net 30", "Net 45", "Net 60"],
    },
    "Koperasi Karyawan Sentosa": {
        "slug": "kopkar_sentosa", "style": "C", "prefix": "KKS-",
        "address": "Kawasan Industri Jababeka Blok C-3, Cikarang", "delivery": "",
        "terms": ["Tempo 14 hari", "Tunai"],
    },
}
UNKNOWN = {
    "Toko Serba Ada Mulia": {
        "slug": "toko_mulia", "style": "D", "prefix": "TSM",
        "address": "Jl. Pasar Baru No. 3, Bogor", "delivery": "", "terms": ["30 hari"],
    }
}


# ---------- number / date formats as printed ----------
def idr_dot(n: int) -> str:  # 1.234.500
    return f"{n:,}".replace(",", ".")


def en_money(n: int) -> str:  # 1,234,500.00
    return f"{n:,.2f}"


def fmt_date(d: date, style: str) -> str:
    return {"A": d.strftime("%d/%m/%Y"), "B": f"{d.day:02d} {MONTHS[d.month - 1]} {d.year}",
            "C": d.isoformat(), "D": d.strftime("%d-%m-%Y")}[style]


def money(n: int, style: str) -> str:
    return {"A": idr_dot(n), "B": en_money(n), "C": "Rp " + idr_dot(n), "D": idr_dot(n)}[style]


# ---------- random PO content ----------
def make_po(rng: random.Random, layout: str, cfg: dict, seq: int, n_items: int | None = None) -> dict:
    po_date = date(2026, 1, 5) + timedelta(days=rng.randint(0, 270))
    deliver = po_date + timedelta(days=rng.choice([3, 5, 7, 10, 14]))
    n_items = n_items or rng.randint(2, 9)
    items = []
    for code, desc, uom, price in rng.sample(CATALOG, n_items):
        qty = rng.choice([5, 10, 12, 20, 24, 25, 30, 40, 50, 60, 100])
        items.append({"code": code, "desc": desc, "uom": uom, "qty": qty, "price": price, "amount": qty * price})
    subtotal = sum(i["amount"] for i in items)
    has_tax = cfg["style"] in ("A", "B")
    tax = round(subtotal * 0.11) if has_tax else 0
    number = {
        "A": f"{cfg['prefix']}{po_date.year}/{po_date.month:02d}/{seq:04d}",
        "B": f"{cfg['prefix']}{po_date.strftime('%y%m')}{seq:03d}",
        "C": f"{cfg['prefix']}{seq:05d}",
        "D": f"{cfg['prefix']}{seq:04d}",
    }[cfg["style"]]
    return {
        "layout": layout, "cfg": cfg, "number": number, "date": po_date, "deliver": deliver,
        "terms": rng.choice(cfg["terms"]), "items": items, "subtotal": subtotal, "tax": tax,
        "total": subtotal + tax, "has_tax": has_tax,
    }


def answer(po: dict, items: list[dict], header: bool, totals: bool) -> dict:
    """Ground truth for one page — values exactly as printed on that page."""
    s = po["cfg"]["style"]
    v: dict = {}
    v["po_number"] = po["number"]
    if header:
        v.update(po_date=fmt_date(po["date"], s), customer_name=po["layout"], customer_address=po["cfg"]["address"])
        if s != "C":
            v["delivery_date"] = fmt_date(po["deliver"], s)
        if po["cfg"]["delivery"]:
            v["delivery_address"] = po["cfg"]["delivery"]
        v["payment_terms"] = po["terms"]
        if s in ("A", "B"):
            v["currency"] = "IDR"
    rows = []
    for it in items:
        row = {"description": it["desc"], "quantity": str(it["qty"]), "unit_price": money(it["price"], s),
               "amount": money(it["amount"], s)}
        if s != "C":
            row["item_code"] = it["code"]
            row["uom"] = it["uom"]
        rows.append(row)
    v["items"] = rows
    if totals:
        if po["has_tax"]:
            v["subtotal"] = money(po["subtotal"], s)
            v["tax_amount"] = money(po["tax"], s)
        v["total_amount"] = money(po["total"], s)
    return v


# ---------- drawing helpers ----------
def txt(page, x, y, s, size=9, bold=False, color=(0, 0, 0)):
    page.insert_text((x, y), s, fontsize=size, fontname="hebo" if bold else "helv", color=color)


def rtxt(page, x_right, y, s, size=9, bold=False):
    w = fitz.get_text_length(s, fontname="hebo" if bold else "helv", fontsize=size)
    txt(page, x_right - w, y, s, size, bold)


def table(page, y, cols, rows, size=8.5, border=True, header_fill=(0.9, 0.92, 0.95)):
    """cols: [(title, x_left, x_right, align)] ; returns y after the table."""
    x0, x1, h = cols[0][1], cols[-1][2], 16
    if border:
        page.draw_rect(fitz.Rect(x0, y, x1, y + h), color=(0, 0, 0), fill=header_fill, width=0.6)
    else:
        page.draw_line((x0, y + h), (x1, y + h), width=0.8)
    for title, l, r, align in cols:
        (rtxt(page, r - 4, y + 11, title, size, True) if align == "r" else txt(page, l + 4, y + 11, title, size, True))
    y += h
    for row in rows:
        if border:
            page.draw_rect(fitz.Rect(x0, y, x1, y + h), color=(0, 0, 0), width=0.4)
        for (title, l, r, align), val in zip(cols, row):
            (rtxt(page, r - 4, y + 11, val, size) if align == "r" else txt(page, l + 4, y + 11, val, size))
        y += h
    if border:
        for _, l, _, _ in cols[1:]:
            page.draw_line((l, y - h * (len(rows) + 1)), (l, y), width=0.4)
    return y


# ---------- layouts ----------
def draw_A(doc, po):
    p = doc.new_page(width=595, height=842)
    s = "A"
    txt(p, 40, 60, po["layout"].upper(), 15, True, (0.05, 0.25, 0.5))
    txt(p, 40, 76, po["cfg"]["address"], 9)
    txt(p, 40, 89, "Telp. (022) 420-1188  |  purchasing@sumbermakmur.example", 8, color=(0.3, 0.3, 0.3))
    rtxt(p, 555, 60, "PURCHASE ORDER", 16, True)
    p.draw_line((40, 100), (555, 100), width=1.2, color=(0.05, 0.25, 0.5))
    info = [("No. PO", po["number"]), ("Tanggal", fmt_date(po["date"], s)),
            ("Tgl. Kirim", fmt_date(po["deliver"], s)), ("Pembayaran", po["terms"]), ("Mata Uang", "IDR")]
    for i, (k, v) in enumerate(info):
        txt(p, 360, 122 + i * 14, k, 9, True)
        txt(p, 430, 122 + i * 14, ": " + v, 9)
    txt(p, 40, 122, "Kepada Yth.", 9, True)
    txt(p, 40, 136, SUPPLIER[0], 9)
    txt(p, 40, 150, SUPPLIER[1], 9)
    txt(p, 40, 178, "Alamat Pengiriman:", 9, True)
    txt(p, 40, 192, po["cfg"]["delivery"], 9)
    cols = [("No", 40, 65, "l"), ("Kode", 65, 125, "l"), ("Nama Barang", 125, 315, "l"), ("Qty", 315, 355, "r"),
            ("Sat", 355, 390, "l"), ("Harga", 390, 465, "r"), ("Jumlah", 465, 555, "r")]
    rows = [[str(i + 1), it["code"], it["desc"], str(it["qty"]), it["uom"], money(it["price"], s), money(it["amount"], s)]
            for i, it in enumerate(po["items"])]
    y = table(p, 215, cols, rows)
    for i, (k, v) in enumerate([("Subtotal", po["subtotal"]), ("PPN 11%", po["tax"]), ("TOTAL", po["total"])]):
        txt(p, 390, y + 18 + i * 15, k, 9, i == 2)
        rtxt(p, 551, y + 18 + i * 15, money(v, s), 9, i == 2)
    txt(p, 40, y + 100, "Hormat kami,", 9)
    txt(p, 40, y + 150, "( Bagian Pembelian )", 9)
    return [answer(po, po["items"], True, True)]


def draw_B(doc, po, per_page=12):
    s = "B"
    chunks = [po["items"][i:i + per_page] for i in range(0, len(po["items"]), per_page)]
    answers, n = [], 0
    for pi, chunk in enumerate(chunks):
        p = doc.new_page(width=612, height=792)
        first, last = pi == 0, pi == len(chunks) - 1
        p.draw_rect(fitz.Rect(0, 0, 612, 70), fill=(0.12, 0.35, 0.22), color=None)
        p.insert_text((40, 32), po["layout"], fontsize=16, fontname="hebo", color=(1, 1, 1))
        p.insert_text((40, 50), po["cfg"]["address"], fontsize=9, fontname="helv", color=(1, 1, 1))
        title = "PURCHASE ORDER" if first else "PURCHASE ORDER (continued)"
        w = fitz.get_text_length(title, fontname="hebo", fontsize=13)
        p.insert_text((572 - w, 40), title, fontsize=13, fontname="hebo", color=(1, 1, 1))
        y = 95
        if first:
            p.draw_rect(fitz.Rect(40, 85, 300, 165), color=(0.6, 0.6, 0.6), width=0.6)
            txt(p, 48, 100, "VENDOR", 8, True, (0.4, 0.4, 0.4))
            txt(p, 48, 115, SUPPLIER[0], 9)
            txt(p, 48, 128, SUPPLIER[1], 9)
            txt(p, 48, 148, "SHIP TO", 8, True, (0.4, 0.4, 0.4))
            txt(p, 48, 160, po["cfg"]["delivery"], 8)
            p.draw_rect(fitz.Rect(320, 85, 572, 165), color=(0.6, 0.6, 0.6), width=0.6)
            for i, (k, v) in enumerate([("PO Number", po["number"]), ("Order Date", fmt_date(po["date"], s)),
                                        ("Required By", fmt_date(po["deliver"], s)), ("Terms", po["terms"]),
                                        ("Currency", "IDR")]):
                txt(p, 328, 100 + i * 14, k, 9, True)
                rtxt(p, 564, 100 + i * 14, v, 9)
            y = 185
        else:
            txt(p, 40, 95, "PO Number:", 9, True)
            txt(p, 100, 95, po["number"], 9)
            y = 110
        cols = [("#", 40, 62, "l"), ("Item Code", 62, 125, "l"), ("Description", 125, 330, "l"), ("Qty", 330, 370, "r"),
                ("Unit", 370, 405, "l"), ("Unit Price", 405, 485, "r"), ("Amount", 485, 572, "r")]
        rows = [[str(n + i + 1), it["code"], it["desc"], str(it["qty"]), it["uom"], en_money(it["price"]), en_money(it["amount"])]
                for i, it in enumerate(chunk)]
        n += len(chunk)
        y = table(p, y, cols, rows, border=False)
        if last:
            for i, (k, v) in enumerate([("Subtotal", po["subtotal"]), ("VAT 11%", po["tax"]), ("Grand Total", po["total"])]):
                txt(p, 405, y + 20 + i * 15, k, 9, i == 2)
                rtxt(p, 568, y + 20 + i * 15, en_money(v), 9, i == 2)
            txt(p, 40, y + 90, "Authorized by: ____________________", 9)
        else:
            txt(p, 40, y + 20, "Continued on next page ...", 8, color=(0.4, 0.4, 0.4))
        rtxt(p, 572, 770, f"Page {pi + 1} of {len(chunks)}", 8)
        a = answer(po, chunk, first, last)
        answers.append(a)
    return answers


def draw_C(doc, po):
    s = "C"
    p = doc.new_page(width=420, height=595)  # A5
    txt(p, 30, 40, po["layout"].upper(), 12, True)
    txt(p, 30, 54, po["cfg"]["address"], 8)
    p.draw_line((30, 62), (390, 62), width=0.8)
    txt(p, 30, 80, "SURAT PESANAN BARANG", 11, True)
    for i, (k, v) in enumerate([("Nomor", po["number"]), ("Tanggal", fmt_date(po["date"], s)), ("Pembayaran", po["terms"])]):
        txt(p, 30, 98 + i * 13, k, 8.5)
        txt(p, 95, 98 + i * 13, ": " + v, 8.5)
    txt(p, 250, 98, "Untuk:", 8.5)
    txt(p, 250, 111, SUPPLIER[0][:30], 8)
    cols = [("Barang", 30, 205, "l"), ("Jml", 205, 240, "r"), ("Harga", 240, 310, "r"), ("Total", 310, 390, "r")]
    rows = [[it["desc"], str(it["qty"]), money(it["price"], s), money(it["amount"], s)] for it in po["items"]]
    y = table(p, 145, cols, rows, size=8, border=False)
    p.draw_line((240, y + 6), (390, y + 6), width=0.6)
    txt(p, 240, y + 20, "Total Bayar", 9, True)
    rtxt(p, 386, y + 20, money(po["total"], s), 9, True)
    txt(p, 30, y + 60, "Pengurus Koperasi", 8)
    return [answer(po, po["items"], True, True)]


def draw_D(doc, po):
    s = "D"
    p = doc.new_page(width=595, height=842)
    p.draw_rect(fitz.Rect(30, 30, 565, 812), color=(0.5, 0.1, 0.1), width=1.5)
    txt(p, 50, 70, "ORDER PEMBELIAN", 18, True, (0.5, 0.1, 0.1))
    txt(p, 50, 92, f"{po['layout']} - {po['cfg']['address']}", 9)
    for i, (k, v) in enumerate([("No.", po["number"]), ("Tgl", fmt_date(po["date"], s)), ("Kirim", fmt_date(po["deliver"], s)),
                                ("Bayar", po["terms"])]):
        txt(p, 50 + i * 130, 125, k, 8, True)
        txt(p, 50 + i * 130, 138, v, 9)
    cols = [("Kode", 50, 110, "l"), ("Uraian", 110, 300, "l"), ("Banyak", 300, 350, "r"), ("Satuan", 350, 395, "l"),
            ("@ Harga", 395, 465, "r"), ("Nilai", 465, 545, "r")]
    rows = [[it["code"], it["desc"], str(it["qty"]), it["uom"], money(it["price"], s), money(it["amount"], s)] for it in po["items"]]
    y = table(p, 165, cols, rows, header_fill=(0.97, 0.9, 0.9))
    txt(p, 395, y + 20, "Jumlah", 10, True)
    rtxt(p, 541, y + 20, money(po["total"], s), 10, True)
    return [answer(po, po["items"], True, True)]


DRAW = {"A": draw_A, "B": draw_B, "C": draw_C, "D": draw_D}


def scan_look(pdf: bytes, rng: random.Random) -> bytes:
    """Render to JPG with slight rotation, blur and noise, like a phone photo / scan."""
    with fitz.open(stream=pdf, filetype="pdf") as d:
        pix = d[0].get_pixmap(dpi=150, colorspace=fitz.csRGB, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    img = img.rotate(rng.uniform(-1.5, 1.5), expand=True, fillcolor=245, resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    noise = Image.effect_noise(img.size, 12).convert("L")
    img = Image.blend(img, noise, 0.06).point(lambda v: min(255, v + 8))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=82)
    return buf.getvalue()


def build(po: dict, rng: random.Random) -> tuple[bytes, str, list[dict]]:
    doc = fitz.open()
    pages = DRAW[po["cfg"]["style"]](doc, po)
    pdf = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    if po["cfg"]["style"] == "C":
        return scan_look(pdf, rng), ".jpg", pages
    return pdf, ".pdf", pages


def generate(out: Path, per_layout_train: int = 15, per_layout_test: int = 3, seed: int = 2026) -> None:
    rng = random.Random(seed)
    for split, layouts, count in (("training", LAYOUTS, per_layout_train), ("extraction_test", {**LAYOUTS, **UNKNOWN}, per_layout_test)):
        key = []
        for li, (layout, cfg) in enumerate(layouts.items()):
            folder = out / split / cfg["slug"]
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                seq = (100 if split == "training" else 900) + i + 1
                # every 4th Berkah Jaya PO is long enough to need 2 pages
                n_items = rng.randint(14, 18) if cfg["style"] == "B" and i % 4 == 3 else None
                po = make_po(rng, layout, cfg, seq, n_items)
                data, ext, pages = build(po, rng)
                name = po["number"].replace("/", "-") + ext
                (folder / name).write_bytes(data)
                key.append({"file": f"{cfg['slug']}/{name}", "customer_layout": layout, "pages": pages})
        (out / split / "answer_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{split}: {len(key)} documents, {sum(len(k['pages']) for k in key)} pages")


if __name__ == "__main__":
    generate(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "practice_docs")
