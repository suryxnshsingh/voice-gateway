import numpy as np
import pytest

from voicegw.turn import SmartTurn, features


def test_features_shape_and_left_padding():
    short = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
    f = features(short)
    assert f.shape == (1, 80, 800) and np.isfinite(f).all()
    p = SmartTurn().p_complete(short)
    assert 0.0 <= p <= 1.0


def test_features_match_whisper_feature_extractor():
    transformers = pytest.importorskip("transformers")  # present in the bench env, not the gateway's
    fe = transformers.WhisperFeatureExtractor(chunk_length=8)
    x = np.random.default_rng(1).normal(0, 0.1, 16000 * 3).astype(np.float32)
    n = 8 * 16000
    padded = np.pad(x, (n - len(x), 0))
    ref = fe(padded, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=n,
             truncation=True, do_normalize=True).input_features
    assert np.abs(features(x) - ref).max() < 1e-4
