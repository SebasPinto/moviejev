from __future__ import annotations

from moviejev.models import TasteProfile


def test_profile_clips_overlong_free_text_instead_of_failing() -> None:
    p = TasteProfile(tone="x" * 500, era="y" * 500)
    assert len(p.tone) == 120 and len(p.era) == 60


def test_profile_clips_overlong_lists() -> None:
    p = TasteProfile(themes=[f"t{i}" for i in range(20)], liked_titles=["a"] * 30)
    assert len(p.themes) == 8 and len(p.liked_titles) == 10
