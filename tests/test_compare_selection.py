"""Production-model selection from held-out comparison rows."""

from compare_models import select_production_model


def _row(name, macro_f1, p95):
    return {"model": name, "macro_f1": macro_f1, "p95_cpu_latency_ms": p95}


def test_lstm_wins_on_macro_f1():
    winner = select_production_model([
        _row("lstm", 0.90, 200),
        _row("gru", 0.80, 100),
        _row("tcn", 0.85, 90),
        _row("mlp", 0.70, 50),
        _row("transformer", 0.89, 300),
    ])
    assert winner["model"] == "lstm"


def test_equal_f1_keeps_the_lower_p95():
    winner = select_production_model([
        _row("lstm", 0.90, 200),
        _row("tcn", 0.90, 120),
    ])
    assert winner["model"] == "tcn"


def test_transformer_stays_an_ablation_when_only_f1_is_better():
    winner = select_production_model([
        _row("lstm", 0.90, 150),
        _row("transformer", 0.95, 400),
    ])
    assert winner["model"] == "lstm"


def test_transformer_wins_only_when_f1_and_p95_are_both_better():
    winner = select_production_model([
        _row("lstm", 0.90, 200),
        _row("transformer", 0.94, 150),
    ])
    assert winner["model"] == "transformer"
