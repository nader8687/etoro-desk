"""Fleet assets resolve only through hardcoded KNOWN_ASSET_LABELS."""
import instrument_config


def test_bitcoin_uses_known_label_only():
    all_labels = {
        "Bitcoin Future CME  (BTC.Fut)": 315,
        "Bitcoin  (BTC)": 100000,
    }
    lbl = instrument_config.label_for_asset_name(
        "Bitcoin", [], all_labels=all_labels,
    )
    assert lbl == "Bitcoin  (BTC)"


def test_dogecoin_uses_known_label_only():
    all_labels = {
        "Dogecoin Futures  (DGOA.FUT)": 999,
        "Dogecoin  (DOGE)": 100001,
    }
    lbl = instrument_config.label_for_asset_name(
        "Dogecoin", [], all_labels=all_labels,
    )
    assert lbl == "Dogecoin  (DOGE)"


def test_ignores_futures_label_on_row():
    lbl = instrument_config.label_for_asset_name(
        "Bitcoin",
        [],
        full_label="Bitcoin Future CME  (BTC.Fut)",
    )
    assert lbl == "Bitcoin  (BTC)"


def test_unknown_asset_returns_none():
    assert instrument_config.label_for_asset_name("Ripple", []) is None


def test_normalize_legacy_amazon_row():
    short = instrument_config.normalize_fleet_asset_short(
        "Amazon.com",
        "Amazon.com Inc  (AMZN)",
    )
    assert short == "Amazon"


def test_asset_short_for_label_round_trip():
    for short, label in instrument_config.KNOWN_ASSET_LABELS.items():
        assert instrument_config.asset_short_for_label(label) == short
