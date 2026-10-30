"""Tiny builders for hand-made transaction frames."""
import pandas as pd

from recon import config
from recon.preprocessing import finalize_frame


def make_frame(source, rows):
    """rows: (date, vendor_raw, reference, amount[, direction]) -> frame in the internal schema."""
    recs = []
    for i, r in enumerate(rows):
        date, vendor, ref, amount = r[:4]
        direction = r[4] if len(r) > 4 else "DEBIT"
        recs.append({"source_row": i, "txn_date": pd.Timestamp(date), "vendor_raw": vendor,
                     "reference": ref, "amount": amount, "direction": direction})
    cols = ["source_row", "txn_date", "vendor_raw", "reference", "amount", "direction"]
    return finalize_frame(pd.DataFrame(recs, columns=cols), source)


def pair(name):
    return next(p for p in config.RECON_PAIRS if p["name"] == name)
