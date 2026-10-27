# Customer PO Extraction dengan Donut

Aplikasi web untuk membaca **Purchase Order (PO) pelanggan** berformat PDF/gambar menjadi **JSON terstruktur**
menggunakan model [Donut](https://huggingface.co/docs/transformers/model_doc/donut) (OCR-free Document
Understanding Transformer). Aplikasi ini memenuhi dokumen *BUC Document Extraction with DONUT*:

| BUC | Aktor | Menu di aplikasi | Output |
|---|---|---|---|
| 1. Prepare Annotated Customer PO Samples | Data Annotator | **1 · Anotasi Sampel** | PO Ground Truth Record [Completed] |
| 2. Train and Validate PO Extraction Model | AI Developer | **2 · Training & Validasi** | Model Validation Sheet, Model Release Record |
| 3. Extract Customer PO Data to JSON | Order Processing Officer | **3 · Ekstraksi PO** | Structured PO Data Record (JSON) |

## Komponen

```
donut-po-extractor/
├── config/po_field_definition.json   # PO Data Field Definition Sheet (field yang diekstrak)
├── app/
│   ├── field_definition.py   # baca sheet, validasi anotasi, token Donut, normalisasi output JSON
│   ├── documents.py          # hanya terima PDF/gambar; PDF -> gambar per halaman (PyMuPDF)
│   ├── ground_truth.py       # BUC 1: sampel, ground truth entry, filing record
│   ├── donut_model.py        # fine-tuning & inferensi Donut (Hugging Face transformers)
│   ├── training.py           # BUC 2: split, training, validasi, rilis model
│   ├── extraction.py         # BUC 3: ekstraksi & penyimpanan PO data
│   ├── main.py               # REST API (FastAPI) + web UI
│   └── cli.py                # perintah baris untuk training di server GPU
├── static/                   # web UI (HTML/JS tanpa build step)
└── tests/                    # pytest, termasuk alur end-to-end dengan model Donut mini
```

Semua data disimpan sebagai file di folder `data/` (bisa diubah dengan `DONUT_PO_DATA_DIR`):

```
data/ground_truth/samples/<SMP-id>/       file asli + pages/page_001.png ...
data/ground_truth/entries/<entry>.json    ground truth entry per halaman (draft / completed)
data/ground_truth/records/<GTR-id>/       PO Ground Truth Record yang di-file (+ images/metadata.jsonl format Donut/HF)
data/models/runs/<RUN-id>/                split, log, model hasil training, validation_sheet.json/.csv
data/models/releases/<REL-id>.json        Model Release Record (current.json = model aktif)
data/extractions/<EXT-id>/                PO yang diekstrak, halaman, extraction.json, po_data.json
```

## Menjalankan di Google Colab (tanpa install, GPU gratis)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/awaludis/Warung/blob/claude/sharp-wright-7pybmh/donut-po-extractor/colab/Donut_PO_Extractor.ipynb)

Buka [`colab/Donut_PO_Extractor.ipynb`](colab/Donut_PO_Extractor.ipynb) di Colab, pilih **Runtime → Change runtime
type → T4 GPU**, lalu jalankan sel dari atas. Notebook menyimpan data di Google Drive dan memberi link untuk membuka UI.
Jika repository private, link di atas tidak bisa dibuka langsung: di Colab pilih **File → Open notebook → GitHub**
(login GitHub) atau unggah file `.ipynb`-nya, dan simpan token GitHub di Colab Secrets sebagai `GITHUB_TOKEN`.

## Instalasi & menjalankan

Python 3.10+ dibutuhkan. GPU (CUDA) sangat disarankan untuk training; inferensi bisa di CPU.

```bash
cd donut-po-extractor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # untuk GPU, pasang torch sesuai CUDA dulu: https://pytorch.org
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Buka `http://localhost:8000`. Isi nama dan pilih peran di pojok kanan atas (dicatat sebagai *created_by / filed_by /
released_by* di setiap record).

Saat training pertama kali, base model `naver-clova-ix/donut-base` (~800 MB) diunduh dari Hugging Face.
Untuk server tanpa internet, unduh dulu lalu set `DONUT_PO_BASE_MODEL=/path/ke/donut-base`.

## Alur kerja

### BUC 1 — Anotasi sampel (Data Annotator)
1. Unggah contoh PO (PDF/gambar) dan isi **customer layout** (nama pelanggan / template PO).
2. PDF otomatis dikonversi menjadi gambar per halaman.
3. Form anotasi dibuat otomatis dari PO Data Field Definition Sheet; isi nilai yang benar **seperti tertulis di PO**.
4. *Simpan draft* atau *Simpan & tandai selesai* (field wajib harus terisi).
5. Setelah setiap layout punya cukup halaman selesai, klik **File Ground Truth Record**.

Aturan yang ditegakkan: hanya field di definition sheet yang boleh dianotasi (field lain ditolak), anotasi
selalu pada gambar halaman, dan filing ditolak bila ada layout yang belum terwakili (minimal
`DONUT_PO_MIN_PER_LAYOUT`, default 2 halaman selesai per layout, agar ada data training dan validasi).

### BUC 2 — Training & validasi (AI Developer)
1. Pilih Ground Truth Record dan parameter, klik **Bagi data & mulai training**.
2. Data dibagi per layout menjadi set training dan validasi (tidak pernah tumpang tindih; dicek ulang sebelum validasi).
3. Donut di-fine-tune pada set training (berjalan di background, log & progres tampil di UI).
4. Model dijalankan pada set validasi dan dibandingkan dengan nilai benar.
5. **Model Validation Sheet** mencatat akurasi per field dan per customer layout (+ unduh CSV, daftar halaman yang berbeda).
6. **Rilis model** — hanya bisa setelah validasi tercatat; opsional batas minimal akurasi. Rilis membuat Model Release Record.

Banner atas menandai **perlu retraining** bila definition sheet berubah atau ada layout pelanggan baru
yang belum ada di model rilis.

Training via CLI (mis. di server GPU):

```bash
python -m app.cli records
python -m app.cli train --record GTR-... --epochs 30 --release
```

### BUC 3 — Ekstraksi PO (Order Processing Officer)
1. Unggah PO pelanggan (PDF/gambar), pilih customer layout (disarankan).
2. Donut membaca setiap halaman langsung dari gambarnya; hasil per halaman digabung (header dari halaman pertama
   yang berisi, line item disambung).
3. JSON yang dihasilkan selalu mengikuti struktur definition sheet (field tak dikenal dibuang, field kosong = `null`,
   field `number` menjadi angka JSON).
4. Periksa/koreksi JSON lalu **Simpan PO data** → Structured PO Data Record; unduh sebagai `PO_<nomor>.json`.

Peringatan ditampilkan bila layout tidak termasuk sampel training model rilis (ekstraksi bisa tidak andal) atau
field wajib tidak ditemukan. Ekstraksi ditolak untuk file selain PDF/gambar dan bila definition sheet berubah
setelah model dilatih.

## Mengubah field yang diekstrak

Edit `config/po_field_definition.json`. Tipe field: `string`, `date`, `number`, dan `list` (baris berulang dengan
sub-field, mis. line item). Setelah sheet berubah, entry lama harus disimpan ulang, record baru di-file, dan model
dilatih ulang — aplikasi menegakkan ini otomatis.

## Konfigurasi (environment variable)

| Variabel | Default | Keterangan |
|---|---|---|
| `DONUT_PO_DATA_DIR` | `./data` | Folder penyimpanan |
| `DONUT_PO_FIELD_DEFINITION` | `config/po_field_definition.json` | PO Data Field Definition Sheet |
| `DONUT_PO_BASE_MODEL` | `naver-clova-ix/donut-base` | Checkpoint awal fine-tuning |
| `DONUT_PO_IMAGE_HEIGHT` / `_WIDTH` | `1280` / `960` | Ukuran input Donut |
| `DONUT_PO_MAX_LENGTH` | `768` | Panjang maksimum token output |
| `DONUT_PO_PDF_DPI` | `200` | Resolusi render PDF |
| `DONUT_PO_MIN_PER_LAYOUT` | `2` | Minimal halaman selesai per layout untuk filing |

## Tips akurasi

* Siapkan minimal ±20–50 halaman per layout pelanggan untuk hasil yang andal; 2 halaman hanya batas teknis.
* Tulis nilai ground truth persis seperti di dokumen (format tanggal/angka) — Donut belajar menyalin teks.
* Gunakan GPU ≥16 GB untuk ukuran 1280×960; turunkan ukuran gambar atau naikkan `grad_accum` bila memori kurang.

## Test

```bash
pip install -r requirements-dev.txt
pytest
```

Test end-to-end membuat model Donut mini secara lokal (tanpa unduhan) untuk menguji seluruh alur BUC 1 → 2 → 3
melalui API.
