"""Edge-weighted sizing must fail OPEN (never zero/crash) and respect bands."""
from __future__ import annotations

import position_sizer as ps


def test_edge_multiplier_unknown_bot_is_neutral():
    assert ps.edge_multiplier("") == 1.0
    assert ps.edge_multiplier("definitely_not_a_real_bot_key") == 1.0


def test_edge_multiplier_failopen_on_bad_state(monkeypatch):
    # Any internal error must degrade to a neutral 1.0, never raise.
    monkeypatch.setattr(ps, "_fleet_rows", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert ps.edge_multiplier("anything") == 1.0


def test_edge_multiplier_band_from_strong_oos(monkeypatch):
    """A strong-OOS plan scales the ticket up; the value stays within the
    documented 0.5x-1.5x band."""
    import instrument_config as ic

    spec = next(iter(ic.load_specs()), None)
    if spec is None:
        return  # no configured bots in this environment — nothing to assert
    import strategies as sm
    sd = sm.display_names().get(spec.strategy, spec.strategy)
    asset = spec.label.split()[0]
    monkeypatch.setattr(ps, "_fleet_rows", lambda: [{
        "Status": "ok", "Strategy": sd, "Asset": asset,
        "Interval": spec.interval, "OOS PF": 3.0, "OOS n": 12,
    }])
    # ensure edge sizing is on regardless of saved settings
    import user_settings
    monkeypatch.setattr(user_settings, "trading_settings",
                        lambda: type("T", (), {"edge_sizing": True})())
    m = ps.edge_multiplier(spec.key)
    assert 0.5 <= m <= 1.5
    assert m >= 1.0   # strong OOS never shrinks the ticket
