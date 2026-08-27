#!/usr/bin/env python3
"""Tkinter front end for the same ``fuzzy_core`` matching library.

Usage
-----
    python fuzzy.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fuzzy_core import match_best, read_table, write_table

READ_ERRORS = (
    ValueError,
    KeyError,
    OSError,
    UnicodeError,
    pd.errors.ParserError,
    pd.errors.EmptyDataError,
)


class MatcherGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fuzzy Matcher")
        self.geometry("600x430")
        self.resizable(False, False)
        self.master_df = None
        self.using_df = None
        self._build_widgets()

    def _build_widgets(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        row0 = ttk.Frame(frm)
        row0.pack(fill="x", pady=4)
        ttk.Button(row0, text="Select MASTER file", command=self._load_master).pack(
            side="left"
        )
        self.lbl_master = ttk.Label(row0, text="no file chosen", width=50, anchor="w")
        self.lbl_master.pack(side="left", padx=6)

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
            df = read_table(Path(path))
        except READ_ERRORS as exc:
            messagebox.showerror("Error", f"Cannot read {role} file:\n{exc}")
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
            result_df = match_best(self.master_df, self.using_df, keys)
            write_table(result_df, Path(save_path))
        except READ_ERRORS as exc:
            messagebox.showerror("Matching error", str(exc))
            return
        messagebox.showinfo("Success", f"File written:\n{save_path}")


if __name__ == "__main__":
    MatcherGUI().mainloop()
