from dataclasses import replace

import v2_extended_pool_eval as extended
import v2_selection_knob_matrix as matrix


def test_configuration_grid_covers_every_requested_reach() -> None:
    configs = extended.configurations()
    assert len(configs) == (
        len(extended.MMR_LAMBDAS) + len(extended.DPP_EXPONENTS)
    ) * len(extended.REACH_VALUES)
    for mode in ("mmr", "dpp"):
        assert {config.reach for config in configs if config.mode == mode} == set(
            extended.REACH_VALUES
        )


def test_full_reach_uses_every_non_seed_active_track() -> None:
    config = matrix.SelectorConfig(mode="mmr", reach=1.0)
    assert matrix.effective_pool_count(matrix.EXPECTED_ACTIVE_TRACKS, config) == 80_323


def test_record_key_distinguishes_mode_control_and_reach() -> None:
    base = matrix.SelectorConfig(mode="mmr", mmr_lambda=0.4, reach=0.02)
    keys = {
        extended.record_key(1, base),
        extended.record_key(2, base),
        extended.record_key(1, replace(base, reach=1.0)),
        extended.record_key(1, replace(base, mmr_lambda=0.6)),
        extended.record_key(1, matrix.SelectorConfig(mode="dpp", reach=0.02)),
    }
    assert len(keys) == 5


def test_listening_packet_spans_small_medium_and_full_pools() -> None:
    chosen = extended.selected_listening_configs()
    assert any(value.endswith("reach-0.02") for value in chosen)
    assert any(value.endswith("reach-0.1") for value in chosen)
    assert any(value.endswith("reach-1") for value in chosen)
    assert any(value.startswith("mmr-") for value in chosen)
    assert any(value.startswith("dpp-") for value in chosen)
