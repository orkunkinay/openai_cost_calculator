"""Tests for the automated pricing-update pipeline (independent of any provider)."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

import pytest

from openai_cost_calculator.catalog import ModelPricing, PriceSet, ProviderPricing, Source
from openai_cost_calculator.catalog.io import dump_provider, load_provider_file, write_provider_file
from openai_cost_calculator.sync.base import FixtureFetcher, SourceError, SourceResult
from openai_cost_calculator.sync.report import overall_status, render_json, render_markdown
from openai_cost_calculator.sync.runner import REVERIFY_DAYS, run_sync, sync_provider
from openai_cost_calculator.sync.text import clean_cell, html_tables, html_text, markdown_tables, parse_money, slugify

D = Decimal
TODAY = date(2026, 9, 23)
SOURCE = Source(id="acme-api", kind="official_api", url="https://acme.example/prices")


def _m(model_id, input_price="1", output_price="2", source="acme-api", **kw):
    return ModelPricing(
        id=model_id,
        prices=(PriceSet(rates={"input": D(input_price), "output": D(output_price)}),),
        source=source,
        **kw,
    )


class FakeSource:
    provider = "acme"
    source = SOURCE

    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def fetch(self, fetcher):
        if self.error:
            raise self.error
        return self.result


def _current(*models, verified_at=date(2026, 9, 1), extra_sources=()):
    return ProviderPricing(provider="acme", sources=(SOURCE, *extra_sources), models=tuple(models), verified_at=verified_at)


def _sync(result, current, **kw):
    return sync_provider(FakeSource(result), current, FixtureFetcher({}), today=TODAY, **kw)


# --------------------------------------------------------------------------- policy


def test_small_price_change_and_new_official_model_are_applied():
    outcome = _sync(SourceResult([_m("a", "1.5", "3"), _m("b"), _m("c")]), _current(_m("a"), _m("b")))
    assert outcome.status == "updated"
    applied = {d.diff.model_id: d.diff.kind for d in outcome.applied}
    assert applied == {"a": "changed", "c": "added"}
    models = outcome.data.models_by_id()
    assert models["a"].prices[0].rates["input"] == D("1.5")
    assert outcome.data.verified_at == TODAY


def test_large_change_needs_review_and_keeps_old_price():
    outcome = _sync(SourceResult([_m("a", "10", "2"), _m("b")]), _current(_m("a"), _m("b")))
    assert outcome.status == "needs_review"
    [decision] = outcome.needs_review
    assert "large change (10.00x)" in decision.reasons[0]
    assert outcome.data.models_by_id()["a"].prices[0].rates["input"] == D("1")


def test_removed_models_are_kept_and_reported():
    outcome = _sync(SourceResult([_m("a"), _m("b"), _m("c")]), _current(_m("a"), _m("b"), _m("c"), _m("gone")))
    assert [d.diff.model_id for d in outcome.needs_review] == ["gone"]
    assert "gone" in outcome.data.models_by_id()


def test_collapsed_extraction_fails_closed():
    current = _current(*[_m(f"m{i}") for i in range(10)])
    outcome = _sync(SourceResult([_m("m0", "9", "9")]), current)
    assert outcome.status == "failed"
    assert "format probably changed" in outcome.error
    assert outcome.data is current


def test_parser_flagged_and_implausible_models_are_not_applied():
    result = SourceResult([_m("a", "1.1"), _m("b", "5000", "5000")])
    result.issue("price cell said 'contact us'", model_id="a")
    outcome = _sync(result, _current(_m("a"), _m("x")))
    reasons = {d.diff.model_id: d.reasons for d in outcome.needs_review}
    assert "parser flagged" in reasons["a"][0]
    assert any("implausibly high" in r for r in reasons["b"])
    assert "b" not in outcome.data.models_by_id()


def test_non_official_sources_cannot_add_models():
    class Aggregator(FakeSource):
        source = Source(id="acme-api", kind="aggregator", url="https://agg.example")

    outcome = sync_provider(Aggregator(SourceResult([_m("a"), _m("new")])), _current(_m("a")), FixtureFetcher({}), today=TODAY)
    assert outcome.needs_review[0].reasons == ["new model from a non-official source"]


def test_corroboration_blocks_changes_an_independent_source_disputes():
    def still_old(provider, model):
        return {"input": D("1")}

    outcome = _sync(SourceResult([_m("a", "1.2"), _m("b")]), _current(_m("a"), _m("b")), corroborate=still_old)
    assert "independent source still reports the old price" in outcome.needs_review[0].reasons[0]

    agrees = _sync(SourceResult([_m("a", "1.2"), _m("b")]), _current(_m("a"), _m("b")), corroborate=lambda p, m: {"input": D("1.2")})
    assert agrees.status == "updated"
    assert "corroborated" in agrees.applied[0].notes[0]


@pytest.mark.parametrize("error", [SourceError("HTTP 503"), KeyError("pricing"), ValueError("bad cell")])
def test_source_and_parser_failures_are_contained(error):
    current = _current(_m("a"))
    outcome = sync_provider(FakeSource(error=error), current, FixtureFetcher({}), today=TODAY)
    assert outcome.status == "failed" and outcome.data is current


def test_manual_entries_are_preserved_until_an_official_source_supersedes_them():
    manual = Source(id="manual", kind="manual", url="https://acme.example/blog")
    current = _current(_m("a"), _m("legacy", source="manual"), _m("promoted", source="manual"), extra_sources=(manual,))
    outcome = _sync(SourceResult([_m("a"), _m("promoted", "3", "4")]), current)
    models = outcome.data.models_by_id()
    assert models["legacy"].source == "manual"
    assert models["promoted"].source == "acme-api"


def test_verified_at_is_restamped_only_when_stale_or_changed():
    fresh = _current(_m("a"), verified_at=date(2026, 9, 20))
    assert _sync(SourceResult([_m("a")]), fresh).data.verified_at == date(2026, 9, 20)
    stale = _current(_m("a"), verified_at=date.fromordinal(TODAY.toordinal() - REVERIFY_DAYS))
    assert _sync(SourceResult([_m("a")]), stale).data.verified_at == TODAY


# --------------------------------------------------------------------------- runner / reports


def test_run_sync_writes_only_real_changes_and_reports(tmp_path):
    write_provider_file(tmp_path / "acme.json", _current(_m("a"), verified_at=date(2026, 9, 20)))
    before = (tmp_path / "acme.json").read_text()

    unchanged = run_sync([FakeSource(SourceResult([_m("a")]))], tmp_path, FixtureFetcher({}), today=TODAY)
    assert unchanged[0].status == "unchanged" and not unchanged[0].written
    assert (tmp_path / "acme.json").read_text() == before

    dry = run_sync([FakeSource(SourceResult([_m("a", "1.2")]))], tmp_path, FixtureFetcher({}), today=TODAY, dry_run=True)
    assert dry[0].status == "updated" and not dry[0].written

    outcomes = run_sync([FakeSource(SourceResult([_m("a", "1.2"), _m("z", "99", "99")]))], tmp_path, FixtureFetcher({}), today=TODAY)
    assert outcomes[0].written
    assert load_provider_file(tmp_path / "acme.json").models_by_id()["a"].prices[0].rates["input"] == D("1.2")

    markdown = render_markdown(outcomes, today=TODAY)
    assert "| acme | updated |" in markdown
    assert "input 1 -> 1.2" in markdown
    report = json.loads(render_json(outcomes, today=TODAY))
    assert report["status"] == "updated"
    assert report["providers"][0]["decisions"][0]["applied"] is True
    assert overall_status([*outcomes, sync_provider(FakeSource(error=SourceError("x")), None, FixtureFetcher({}), today=TODAY)]) == "failed"


def test_fixture_fetcher_reports_missing_fixture():
    with pytest.raises(SourceError, match="no fixture"):
        FixtureFetcher({}).get_text("https://missing.example")


# --------------------------------------------------------------------------- text helpers


@pytest.mark.parametrize(
    "cell,value",
    [("$1.25", D("1.25")), ("\\$0.075", D("0.075")), ("$3 / MTok", D("3")), ("Free", D(0)), ("-", None), ("—", None), ("$1,000.50", D("1000.50"))],
)
def test_parse_money(cell, value):
    assert parse_money(cell) == value


@pytest.mark.parametrize("cell", ["$0.30 (text) $1.00 (audio)", "about $2", "$0.111 per hour"])
def test_parse_money_rejects_anything_ambiguous(cell):
    with pytest.raises(ValueError):
        parse_money(cell)


def test_markdown_tables_keep_heading_context_and_escaped_pipes():
    text = "# Pricing\n## Batch\nPrices per 1M tokens.\n| Model | Input |\n| --- | ---: |\n| [gpt-x](/m) \\| mini | \\$1.00 |\n\ntrailing"
    [table] = markdown_tables(text)
    assert table.headings == ["Pricing", "Batch"]
    assert table.preamble == ["Prices per 1M tokens."]
    assert table.header == ["Model", "Input"]
    assert [clean_cell(c) for c in table.rows[0]] == ["gpt-x | mini", "$1.00"]
    assert table.column("input") == 1


def test_html_tables_and_text():
    doc = "<h2>Claude</h2><table><tr><th>Model</th><th>Price</th></tr><tr><td>Sonnet<br>4.5</td><td>$3</td></tr></table><script>x=1</script><p>Note</p>"
    [table] = html_tables(doc)
    assert table.headings == ["Claude"]
    assert table.header == ["Model", "Price"] and table.rows == [["Sonnet 4.5", "$3"]]
    assert html_text(doc).splitlines()[-1] == "Note"


def test_slugify():
    assert slugify("Claude Sonnet 4.5") == "claude-sonnet-4-5"
