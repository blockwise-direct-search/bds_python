from __future__ import annotations

from tests.profile_optiprofiler import _build_benchmark_options, _parse_feature_name


def test_noisy_feature_display_name_keeps_noise_level():
    feature_name, feature_options, display_name = _parse_feature_name("noisy_1e-4")

    benchmark_options = _build_benchmark_options(
        {"plibs": "s2mpj", "savepath": "/tmp"},
        feature_name,
        feature_options,
        ["nelder-mead", "powell"],
    )

    assert feature_name == "noisy"
    assert display_name == "noisy_1e-4"
    assert benchmark_options["feature_stamp"] == "noisy_0.0001_mixed_gaussian_s2mpj"


def test_linearly_transformed_noisy_feature_display_name_hides_custom():
    feature_name, feature_options, display_name = _parse_feature_name("linearly_transformed_noisy_1e-4")

    benchmark_options = _build_benchmark_options(
        {"plibs": "s2mpj", "savepath": "/tmp"},
        feature_name,
        feature_options,
        ["nelder-mead", "powell"],
    )

    assert feature_name == "custom"
    assert display_name == "linearly_transformed_noisy_1e-4"
    assert benchmark_options["feature_stamp"] == "linearly_transformed_noisy_1e-4_s2mpj"
