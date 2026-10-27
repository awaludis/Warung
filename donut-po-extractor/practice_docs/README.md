# Dokumen latihan (PO fiktif)

Semua perusahaan, alamat, nomor, dan nilai di sini **fiktif**, dibuat oleh `tools/generate_practice_docs.py`.

| Customer layout | Format | Ciri khas | Training | Uji ekstraksi |
|---|---|---|---|---|
| PT Sumber Makmur Sejahtera | PDF | Bahasa Indonesia, tabel bergaris, tanggal `16/02/2026`, angka `1.250.000`, PPN 11% | 15 | 3 |
| CV Berkah Jaya Abadi | PDF | Bahasa Inggris, header hijau, tanggal `14 Jul 2026`, angka `1,250,000.00`; ada PO **2 halaman** | 15 (18 hal.) | 3 |
| Koperasi Karyawan Sentosa | JPG | Kertas A5 hasil "scan" (miring, sedikit blur), tanggal `2026-04-02`, angka `Rp 585.000`, tanpa kode barang & pajak | 15 | 3 |
| Toko Serba Ada Mulia | PDF | Layout **tidak ada** di data training — untuk melihat peringatan BUC 3 | – | 3 |

```
practice_docs/
├── training/            # untuk BUC 1 (anotasi) dan BUC 2 (training)
│   ├── answer_key.json  # nilai benar per halaman, persis seperti tertulis
│   ├── sumber_makmur/  berkah_jaya/  kopkar_sentosa/
└── extraction_test/     # untuk BUC 3 — jangan dipakai untuk training
    ├── answer_key.json
    └── sumber_makmur/  berkah_jaya/  kopkar_sentosa/  toko_mulia/
```

## Cara latihan

**A. Latihan anotasi (BUC 1 manual)** — unggah beberapa file dari `training/` di Tab 1, isi field-nya sendiri,
lalu bandingkan dengan `training/answer_key.json`.

**B. Langsung ke training** — impor semua dokumen training beserta kunci jawabannya sebagai ground truth entry:

```bash
python -m app.cli import-samples practice_docs/training          # entry langsung "selesai"
python -m app.cli import-samples practice_docs/training --draft  # atau sebagai draft untuk diperiksa di UI
```

Lalu di Tab 1 klik **File Ground Truth Record**, dan lanjut training di Tab 2.
(Impor sekali saja — mengimpor dua kali membuat sampel ganda.)

**C. Uji ekstraksi (BUC 3)** — setelah model dirilis, unggah file dari `extraction_test/` di Tab 3 dan bandingkan
hasilnya dengan `extraction_test/answer_key.json`. File `toko_mulia/` akan memunculkan peringatan layout tidak dikenal.

## Catatan

* 45 dokumen cukup untuk mencoba alur end-to-end. Akurasi bagus biasanya butuh lebih banyak sampel per layout;
  buat lebih banyak dengan mengubah `per_layout_train` di `tools/generate_practice_docs.py`.
* Di kunci jawaban, angka ditulis seperti tercetak (`1.250.000`, `1,250,000.00`, `Rp 585.000`); hasil ekstraksi
  mengubahnya menjadi angka JSON (`1250000`).
* Untuk PO dua halaman, halaman 1 berisi header + baris barang, halaman 2 berisi nomor PO + sisa barang + total.
