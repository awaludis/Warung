"""Customer PO extraction with Donut (BUC 1-3)."""

# Use the operating system's certificate store for HTTPS (downloading the Donut base model from
# Hugging Face). Without this, Python rejects certificates from company proxies / antivirus HTTPS
# scanning that Windows and macOS already trust ("CERTIFICATE_VERIFY_FAILED").
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001 - optional; plain certifi is used when unavailable
    pass
