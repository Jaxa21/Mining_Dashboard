import re
from pathlib import Path

import pandas as pd


class DataSource:
    SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample-data.csv"
    CSV_URL = "https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet=Data"
    SHEET_ID = re.compile(r"/spreadsheets/d/([\w-]+)")
    MIN_DAYS = 7

    def load(self, link: str) -> pd.DataFrame:
        source = self.csv_url(link) if link.strip() else self.SAMPLE
        try:
            raw = pd.read_csv(source)
        except (OSError, ValueError) as error:
            raise ValueError("Could not read the sheet. Is it shared as 'Anyone with the link'?") from error
        return self.prepare(raw)

    @classmethod
    def csv_url(cls, link: str) -> str:
        match = cls.SHEET_ID.search(link)
        if not match:
            raise ValueError("This does not look like a Google Sheets link.")
        return cls.CSV_URL.format(match.group(1))

    def prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.dropna(subset=[raw.columns[0]])
        mines = [column for column in raw.columns[1:] if not str(column).startswith("Unnamed")]
        data = raw[mines].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").fillna(0)
        if data.empty or len(data) < self.MIN_DAYS:
            raise ValueError(f"The sheet needs at least one mine and {self.MIN_DAYS} days of data.")
        data.index = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0]), name="Date")
        data["Total"] = data.sum(axis=1).round(1)
        return data.sort_index()
