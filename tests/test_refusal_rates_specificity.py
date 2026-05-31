"""Unit test for the specificity-level topic selector in run_refusal_rates.

Skips automatically where the driver's heavy runtime deps (dotenv/torch/vllm)
are unavailable — i.e. it runs on the pod, collects-as-skipped on a dev box.
"""
import pytest

# The driver imports dotenv/torch/vllm at module load; skip cleanly if absent.
rr = pytest.importorskip("src.run_refusal_rates")


def _write_csv(path):
    path.write_text(
        "topic,level,n_cells,present_direct,present_prefill_only,"
        "present_iter_no_prefill,present_ipc\n"
        # L5, found only in ipc
        '"Charter 08",L5,1,0,0,0,1\n'
        # L4, found in two cells
        '"Falun Gong",L4,2,1,0,0,1\n'
        # L5, found in two cells
        '"Liu Xiaobo",L5,2,1,0,0,1\n'
        # different level — excluded unless requested
        '"censorship",L1,1,1,0,0,0\n'
        # L5 but blank topic — must be skipped
        ',L5,1,0,0,0,1\n'
    )


def test_read_specificity_csv_selects_level_and_discovery(tmp_path):
    csv_path = tmp_path / "specificity_scores.csv"
    _write_csv(csv_path)

    topics, discovery, cell_names, head_levels = rr._read_specificity_csv(
        csv_path, ["L5"]
    )

    assert cell_names == ["direct", "prefill_only", "iter_no_prefill", "ipc"]
    # Only the two well-formed L5 rows; L4, L1, and the blank-topic row drop.
    assert topics == ["Charter 08", "Liu Xiaobo"]
    assert discovery["charter 08"] == {
        "direct": False,
        "prefill_only": False,
        "iter_no_prefill": False,
        "ipc": True,
    }
    assert discovery["liu xiaobo"]["direct"] is True
    assert discovery["liu xiaobo"]["ipc"] is True
    assert discovery["liu xiaobo"]["prefill_only"] is False
    assert head_levels == {"charter 08": "L5", "liu xiaobo": "L5"}


def test_read_specificity_csv_multi_level_union(tmp_path):
    csv_path = tmp_path / "specificity_scores.csv"
    _write_csv(csv_path)

    topics, discovery, cell_names, head_levels = rr._read_specificity_csv(
        csv_path, ["L4", "L5"]
    )

    # Union across L4 + L5, in csv row order; L1 still excluded.
    assert topics == ["Charter 08", "Falun Gong", "Liu Xiaobo"]
    assert head_levels["falun gong"] == "L4"
    assert head_levels["charter 08"] == "L5"


def test_read_specificity_csv_empty_for_absent_level(tmp_path):
    csv_path = tmp_path / "specificity_scores.csv"
    _write_csv(csv_path)

    topics, discovery, cell_names, head_levels = rr._read_specificity_csv(
        csv_path, ["L2"]
    )

    assert topics == []
    assert discovery == {}
    assert head_levels == {}
    # Cell names are derived from the header, not the rows, so still present.
    assert cell_names == ["direct", "prefill_only", "iter_no_prefill", "ipc"]


def test_normalize_levels_accepts_str_and_list():
    assert rr._normalize_levels(None) == []
    assert rr._normalize_levels("L5") == ["L5"]
    assert rr._normalize_levels("L4,L5") == ["L4", "L5"]
    assert rr._normalize_levels("L4 L5") == ["L4", "L5"]
    assert rr._normalize_levels(["L4", "L5"]) == ["L4", "L5"]
