"""Automated, fail-closed pricing updates from authoritative upstream sources.

Pipeline per provider: fetch (official API or documentation page) -> parse into
catalog entries -> validate -> diff against the checked-in data -> apply only
changes that pass every guard in :mod:`.policy` -> write deterministic JSON and
a report.  See ``docs/PRICING_UPDATES.md``.
"""
