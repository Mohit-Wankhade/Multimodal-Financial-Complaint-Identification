from src.fincimm.preprocessing import clean_text, parse_web_entities, split_labels


def test_clean_text():
    assert clean_text("Payment FAILED!!!") == "payment failed"


def test_parse_web_entities():
    assert parse_web_entities("payment|upi|bank") == ["payment", "upi", "bank"]


def test_split_labels():
    assert split_labels("Payment Failure|Delay Response") == ["net_banking_issue", "provider_response"]
