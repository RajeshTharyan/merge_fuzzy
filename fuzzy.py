#!/usr/bin/env python3
"""
Fuzzy-Matching GUI Helper
=========================
A minimal Tkinter front-end for fuzzy matching two datasets.

Change log (2025-05-08-d)
------------------------
* **Save-as options** – user can now choose **one** output format (CSV, Excel
  *.xlsx*, or Stata *.dta*) in the *Save As* dialog. Only that format is
  written.
* Retains: multi-engine fuzzy matching, footer credit, dynamic key list.

Usage
-----
    python fuzzy_matcher_gui.py
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from rapidfuzz import fuzz, process
import textdistance
import recordlinkage as rl


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
def _read_file(path: Path) -> pd.DataFrame:
    """Load CSV, Excel, or Stata .dta into a DataFrame."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".dta":
        return pd.read_stata(path)
    raise ValueError(f"Unsupported file type: {ext}")


def _write_output(df: pd.DataFrame, path: Path) -> None:
    """Save *df* according to the extension in *path*."""
    ext = path.suffix.lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".xlsx":
        df.to_excel(path, index=False)
    elif ext == ".dta":
        df.to_stata(path, write_index=False)
    else:
        raise ValueError(f"Unsupported output format: {ext}")


def _validate_keys(df: pd.DataFrame, keys: List[str]) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError("Missing key column(s): " + ", ".join(missing))
    for k in keys:
        if not pd.api.types.is_string_dtype(df[k]):
            df[k] = df[k].astype(str)


def _build_key_series(df: pd.DataFrame, keys: List[str]) -> pd.Series:
    return (
        df[keys]
        .fillna("")
        .apply(lambda r: " ".join(r.astype(str)), axis=1)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Matching engines
# ─────────────────────────────────────────────────────────────────────────────
def _best_match_rapidfuzz(target: str, universe: pd.Series):
    match, score, idx = process.extractOne(
        target, universe, scorer=fuzz.token_sort_ratio
    )
    return universe.index[idx], score, "rapidfuzz"


def _best_match_textdistance(target: str, universe: pd.Series):
    sims = universe.map(
        lambda x: textdistance.jaro_winkler.normalized_similarity(target, x)
    )
    idx = sims.idxmax()
    return idx, sims.loc[idx] * 100, "textdistance"


def _best_match_recordlinkage(
    i: int, master_keys: pd.Series, using_keys: pd.Series
):
    master_single = master_keys.iloc[[i]].to_frame(name="key")
    using_df = using_keys.to_frame(name="key")

    idxer = rl.index.Full()
    pairs = idxer.index(master_single, using_df)

    compare = rl.Compare()
    compare.string("key", "key", method="jaro", label="jw")
    scores_df = compare.compute(pairs, master_single, using_df)
    scores = scores_df["jw"]
    if scores.empty:
        return pd.NA, 0.0, "recordlinkage"
    best_pair = scores.idxmax()
    return best_pair[1], scores.loc[best_pair] * 100, "recordlinkage"


# ─────────────────────────────────────────────────────────────────────────────
# Core matcher
# ─────────────────────────────────────────────────────────────────────────────
def fuzzy_match(master_df: pd.DataFrame, using_df: pd.DataFrame, keys: List[str]):
    _validate_keys(master_df, keys)
    _validate_keys(using_df, keys)

    master_keys = _build_key_series(master_df, keys)
    using_keys = _build_key_series(using_df, keys)

    records = []
    for i, key_string in master_keys.items():
        scores = [
            _best_match_rapidfuzz(key_string, using_keys),
            _best_match_textdistance(key_string, using_keys),
            _best_match_recordlinkage(i, master_keys, using_keys),
        ]
        using_idx, best_score, method = max(scores, key=lambda x: x[1])
        records.append(
            {
                "master_index": i,
                "using_index": using_idx,
                "best_score": round(best_score, 2),
                "method": method,
            }
        )

    link_df = pd.DataFrame(records).set_index("master_index")
    merged = master_df.join(link_df, how="left")
    merged = merged.merge(
        using_df.add_prefix("using_"),
        left_on="using_index",
        right_index=True,
        how="left",
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Tkinter GUI
# ─────────────────────────────────────────────────────────────────────────────
class MatcherGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fuzzy Matcher")
        self.geometry("600x430")
        self.resizable(False, False)
        self.master_df = None
        self.using_df = None
        self._build_widgets()

    # ------------------------------------------------------------------
    # Build GUI
    # ------------------------------------------------------------------
    def _build_widgets(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        # MASTER file
        row0 = ttk.Frame(frm)
        row0.pack(fill="x", pady=4)
        ttk.Button(row0, text="Select MASTER file", command=self._load_master).pack(
            side="left"
        )
        self.lbl_master = ttk.Label(row0, text="no file chosen", width=50, anchor="w")
        self.lbl_master.pack(side="left", padx=6)

        # USING file
        row1 = ttk.Frame(frm)
        row1.pack(fill="x", pady=4)
        ttk.Button(row1, text="Select USING file", command=self._load_using).pack(
            side="left"
        )
        self.lbl_using = ttk.Label(row1, text="no file chosen", width=50, anchor="w")
        self.lbl_using.pack(side="left", padx=6)

        ttk.Separator(frm).pack(fill="x", pady=8)
        ttk.Label(
            frm, text="Select key variable(s): (Ctrl+Click for multiple)"
        ).pack(anchor="w")
        self.lst_keys = tk.Listbox(frm, selectmode="multiple", height=10)
        self.lst_keys.pack(fill="both", expand=True, pady=4)

        ttk.Button(frm, text="Run Matching", command=self._run_matching).pack(pady=8)

        ttk.Label(
            frm,
            text="By: Prof. Rajesh Tharyan",
            font=("TkDefaultFont", 8, "italic"),
        ).pack(side="bottom", pady=(6, 0))

    # ------------------------------------------------------------------
    # File loaders
    # ------------------------------------------------------------------
    def _load_master(self):
        self._load_file("MASTER")

    def _load_using(self):
        self._load_file("USING")

    def _load_file(self, role: str):
        path = filedialog.askopenfilename(
            title=f"Select {role} file",
            filetypes=[
                ("Data files", "*.csv *.xlsx *.xls *.dta"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            df = _read_file(Path(path))
        except Exception as e:
            messagebox.showerror("Error", f"Cannot read {role} file:\n{e}")
            return
        if role == "MASTER":
            self.master_df = df
            self.lbl_master.config(text=Path(path).name)
        else:
            self.using_df = df
            self.lbl_using.config(text=Path(path).name)
        self._populate_keys()

    def _populate_keys(self):
        if self.master_df is None or self.using_df is None:
            return
        common = [c for c in self.master_df.columns if c in self.using_df.columns]
        self.lst_keys.delete(0, tk.END)
        for col in common:
            self.lst_keys.insert(tk.END, col)

    # ------------------------------------------------------------------
    # Run matching and save
    # ------------------------------------------------------------------
    def _run_matching(self):
        if self.master_df is None or self.using_df is None:
            messagebox.showwarning(
                "Missing files", "Load both MASTER and USING files first."
            )
            return
        keys = [self.lst_keys.get(i) for i in self.lst_keys.curselection()]
        if not keys:
            messagebox.showwarning(
                "No variables", "Select at least one key variable to match on."
            )
            return
        save_path = filedialog.asksaveasfilename(
            title="Save output as…",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"), ("Stata", "*.dta")],
        )
        if not save_path:
            return
        try:
            result_df = fuzzy_match(self.master_df, self.using_df, keys)
            _write_output(result_df, Path(save_path))
        except Exception as exc:
            messagebox.showerror("Matching error", str(exc))
            return
        messagebox.showinfo("Success", f"File written:\n{save_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MatcherGUI().mainloop()
