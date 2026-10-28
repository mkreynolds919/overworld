from overworld.graph.games_table import build_games_table, flatten_game, release_year

GTA = {
    "id": 1020,
    "name": "Grand Theft Auto V",
    "first_release_date": 1379376000,
    "game_type": 0,
    "franchises": [{"id": 493, "name": "Grand Theft Auto"}],
    "collections": [{"id": 847, "name": "Grand Theft Auto"}],
    "involved_companies": [
        {"company": {"name": "Take-Two Interactive"}, "developer": False},
        {"company": {"name": "Rockstar North"}, "developer": True},
    ],
    "genres": [{"name": "Shooter"}, {"name": "Racing"}],
    "cover": {"image_id": "co2lbd"},
}


def test_flatten_dedupes_series_and_keeps_only_developers():
    row = flatten_game(GTA)
    assert row["series"] == ["grand theft auto"]
    assert row["developers"] == ["Rockstar North"]
    assert row["display_name"] == "Grand Theft Auto V (2013)"
    assert row["cover_image_id"] == "co2lbd"


def test_flatten_handles_missing_fields():
    row = flatten_game({"id": 1, "name": "Bare Game"})
    assert row["display_name"] == "Bare Game"
    assert row["series"] == [] and row["developers"] == [] and row["themes"] == []
    assert row["cover_image_id"] is None and row["year"] is None


def test_release_year_uses_utc():
    assert release_year(1704067200) == 2024  # 2024-01-01 00:00 UTC


def test_table_drops_duplicate_ids():
    df = build_games_table([GTA, GTA, {"id": 2, "name": "Other"}])
    assert len(df) == 2
    assert df.loc[1020, "name"] == "Grand Theft Auto V"


from overworld.graph.games_table import normalize_series


def test_normalize_series_merges_punctuation_variants():
    assert normalize_series("Vampire: The Masquerade") == normalize_series("vampire the masquerade")
    assert normalize_series("Vampire: The Masquerade – Bloodlines") == "vampire the masquerade bloodlines"
    assert normalize_series("Dungeons & Dragons") == "dungeons dragons"