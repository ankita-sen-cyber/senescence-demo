"""Smoke test — every non-Geneformer module imports cleanly."""


def test_top_level():
    import rnaseq_loop
    assert rnaseq_loop.__version__


def test_scoring():
    from rnaseq_loop.scoring import ScoringConfig, rank_targets, zscore  # noqa


def test_active():
    from rnaseq_loop.active import acquire, bald  # noqa


def test_failure():
    from rnaseq_loop.failure import slice_metrics, counterfactual_probe  # noqa
