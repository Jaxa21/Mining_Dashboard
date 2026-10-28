from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class AnomalyDetector:
    TESTS = ("IQR rule", "Z-score", "Moving average", "Grubbs' test")
    LEVEL_WINDOW = 21

    iqr_k: float = 1.5
    z_threshold: float = 3.0
    ma_window: int = 7
    ma_percent: float = 35.0
    grubbs_alpha: float = 0.05
    adjusted: bool = True

    def flags(self, values: pd.Series) -> pd.DataFrame:
        factor, level = self.baseline(values)
        x, deviation = values / factor, values / factor / level - 1
        flags = pd.DataFrame({name: test(deviation) for name, test in
                              [("IQR rule", self.iqr), ("Z-score", self.zscore), ("Grubbs' test", self.grubbs)]})
        average = self.moving_average(x.mask(flags.any(axis=1)))
        flags.insert(2, "Moving average", (x - average).abs() > average * self.ma_percent / 100)
        flags["Anomaly"] = flags.any(axis=1)
        flags["Expected"] = level * factor
        flags["Type"] = np.where(values >= flags["Expected"], "Spike", "Drop")
        return flags

    def baseline(self, values: pd.Series) -> tuple:
        if not self.adjusted:
            return 1.0, values.median()
        weekday = values.index.dayofweek
        factors = (values / self._moving_median(values)).groupby(weekday).median()
        factor = pd.Series(weekday.map(factors / factors.mean()), index=values.index)
        return factor, self._moving_median(values / factor)

    def moving_average(self, x: pd.Series) -> pd.Series:
        return x.shift(1).rolling(self.ma_window, min_periods=1).mean().ffill().bfill()

    def _moving_median(self, values: pd.Series) -> pd.Series:
        return values.rolling(self.LEVEL_WINDOW, center=True, min_periods=1).median()

    def iqr(self, x: pd.Series) -> pd.Series:
        q1, q3 = x.quantile([0.25, 0.75])
        return (x < q1 - self.iqr_k * (q3 - q1)) | (x > q3 + self.iqr_k * (q3 - q1))

    def zscore(self, x: pd.Series) -> pd.Series:
        return ((x - x.mean()) / x.std()).abs() > self.z_threshold

    def grubbs(self, x: pd.Series) -> pd.Series:
        remaining, outliers = x.dropna(), pd.Series(False, index=x.index)
        while len(remaining) > 6 and outliers.sum() < len(x) / 4:
            distance = (remaining - remaining.mean()).abs()
            day = distance.idxmax()
            if distance[day] <= self.critical_value(len(remaining)) * remaining.std():
                break
            outliers[day] = True
            remaining = remaining.drop(day)
        return outliers

    def critical_value(self, n: int) -> float:
        t = stats.t.ppf(1 - self.grubbs_alpha / (2 * n), n - 2)
        return (n - 1) / np.sqrt(n) * np.sqrt(t**2 / (n - 2 + t**2))
