from predict_weather.data_model import to_dataframe
from predict_weather.synthetic import build_synthetic_dataset


def test_synthetic_is_deterministic():
    a = build_synthetic_dataset(n=50, rng_seed=123)
    b = build_synthetic_dataset(n=50, rng_seed=123)
    assert [o.market_id for o in a] == [o.market_id for o in b]
    assert all(x.poly_prob == y.poly_prob for x, y in zip(a, b))


def test_synthetic_fields_in_range():
    obs = build_synthetic_dataset(n=80)
    df = to_dataframe(obs)
    assert df["poly_prob"].between(0, 1).all()
    assert df["noaa_prob"].between(0, 1).all()
    assert df["outcome"].isin([0, 1]).all()
    assert df["city"].isin(
        list(__import__("src.predict_weather.data_model", fromlist=["CITY_COORDS"]).CITY_COORDS.keys())
    ).all()
