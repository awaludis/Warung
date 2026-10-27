import pytest

from app.field_definition import SEP_TOKEN, load_definition, parse_definition, parse_number
from app.storage import BusinessRuleError


@pytest.fixture
def fd():
    return load_definition()


def test_special_tokens_cover_all_fields(fd):
    toks = fd.special_tokens()
    assert fd.task_token in toks and SEP_TOKEN in toks
    for name in ("po_number", "items", "quantity", "total_amount"):
        assert f"<s_{name}>" in toks and f"</s_{name}>" in toks
    assert len(toks) == len(set(toks))


def test_token_sequence(fd):
    seq = fd.to_token_sequence(
        {"po_number": "PO-1", "items": [{"description": "A", "quantity": "2"}, {"description": "B", "quantity": "3"}]}
    )
    assert seq == (
        "<s_po_number>PO-1</s_po_number><s_items><s_description>A</s_description><s_quantity>2</s_quantity>"
        "<sep/><s_description>B</s_description><s_quantity>3</s_quantity></s_items>"
    )


def test_annotation_rejects_unknown_fields(fd):
    with pytest.raises(BusinessRuleError, match="not in the PO Data Field Definition Sheet"):
        fd.validate_annotation({"po_number": "1", "salesman": "x"})
    with pytest.raises(BusinessRuleError, match="row 1"):
        fd.validate_annotation({"items": [{"description": "x", "colour": "red"}]})


def test_annotation_checks_numbers_and_cleans(fd):
    with pytest.raises(BusinessRuleError, match="must be a number"):
        fd.validate_annotation({"total_amount": "abc"})
    clean = fd.validate_annotation({"po_number": "  PO  9 ", "items": [{"description": "", "quantity": ""}]})
    assert clean["po_number"] == "PO 9"
    assert clean["items"] == []  # empty rows dropped
    assert set(clean) == {f.name for f in fd.fields}


def test_missing_required(fd):
    missing = fd.missing_required(fd.validate_annotation({"po_number": "1", "items": [{"description": "x"}]}))
    assert "po_date" in missing and "items[1].quantity" in missing and "po_number" not in missing


def test_normalize_output_follows_sheet(fd):
    out = fd.normalize_output(
        {"po_number": "PO-1", "junk": "x", "items": {"description": "A", "quantity": "1.000", "bogus": 1}, "total_amount": "Rp 12.500"}
    )
    assert set(out) == {f.name for f in fd.fields}
    assert out["items"] == [{"item_code": None, "description": "A", "quantity": 1000, "uom": None, "unit_price": None, "amount": None}]
    assert out["total_amount"] == 12500 and out["po_date"] is None


@pytest.mark.parametrize(
    "text,expected",
    [("1,234.56", 1234.56), ("1.234,56", 1234.56), ("12.500", 12500), ("2,5", 2.5), ("0.125", 0.125),
     ("1.000.000", 1000000), ("Rp 15.000", 15000), ("-3", -3), ("abc", None), ("10", 10)],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


def test_definition_validation():
    with pytest.raises(ValueError):
        parse_definition({"fields": [{"name": "Bad Name"}]})
    with pytest.raises(ValueError):
        parse_definition({"fields": [{"name": "a", "type": "list"}]})
    a = parse_definition({"fields": [{"name": "a"}]})
    b = parse_definition({"fields": [{"name": "a"}, {"name": "b"}]})
    assert a.hash != b.hash
