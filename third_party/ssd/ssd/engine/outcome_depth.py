"""Deterministic continuation budgets; fan-out and target arithmetic stay fixed."""
import numpy as np


STAT_NAMES = ('trees', 'computed_query_rows', 'dense_query_rows',
              'rank0_hits', 'rank1_hits', 'rank2_hits', 'short_cache_responses',
              'higher_rank_hits')


def validate_depths(depths, maximum, fanout):
    if len(depths) != fanout or any(type(d) is not int or not 1 <= d <= maximum for d in depths):
        raise ValueError('One integer continuation depth in [1,K] is required per recovery rank')


def row_depths(maximum, fanout, depths):
    validate_depths(depths, maximum, fanout)
    return tuple(depths) * (maximum + 1)


def active_rows_by_depth(maximum, fanout, depths):
    lengths = row_depths(maximum, fanout, depths)
    return tuple(tuple(i for i, length in enumerate(lengths) if length > step)
                 for step in range(max(lengths)))


def compact_mask(maximum, fanout, step, prefix_length, active_rows):
    """Mask compact queries against canonical, sparsely written tree KV slots.

    Query i sees the original committed prefix, glue through its own parent,
    and its own ancestor row at every preceding depth. It never sees another
    alternative or an unwritten retired row. KV addresses do not get repacked.
    """
    width = fanout * (maximum + 1)
    rows = np.asarray(active_rows, dtype=np.int64)
    if (prefix_length < 0 or step < 0 or step >= maximum or rows.size == 0 or
            np.any(rows < 0) or np.any(rows >= width) or len(set(rows.tolist())) != len(rows)):
        raise ValueError('Invalid compact tree mask geometry')
    cols = prefix_length + maximum + 1 + (step + 1) * width
    mask = np.zeros((len(rows), cols), dtype=np.uint8)
    mask[:, :prefix_length] = 1
    mask[:, prefix_length:prefix_length+maximum+1] = (
        np.arange(maximum+1)[None, :] <= (rows // fanout)[:, None])
    for ancestor in range(step + 1):
        mask[np.arange(len(rows)), prefix_length+maximum+1+ancestor*width+rows] = 1
    return mask
