"""
src/utils/logging_utils.py
Lightweight, crash-safe episode logging. Writes one row per episode
immediately (not buffered in memory until the end), so a crash partway
through a long run doesn't lose everything before it.
"""

import os
import csv


class IncrementalCSVLogger:
    """
    Appends one row per episode to a CSV file immediately after it's
    computed. Writes the header on first row. Safe to interrupt (Ctrl+C
    or crash) at any point -- all rows written so far are on disk.
    """

    def __init__(self, output_path, fieldnames):
        self.output_path = output_path
        self.fieldnames = fieldnames
        self._header_written = os.path.exists(output_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._file = open(output_path, 'a', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)

        if not self._header_written:
            self._writer.writeheader()
            self._file.flush()

    def log(self, row_dict):
        """Write one episode's data immediately and flush to disk."""
        clean_row = {k: row_dict.get(k, '') for k in self.fieldnames}
        self._writer.writerow(clean_row)
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
