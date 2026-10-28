import pandas as pd

from mining.anomaly_detector import AnomalyDetector


class Analysis:
    COLUMNS = ["Series", "Type", "Start", "End", "Days", "Peak day", "Value", "Expected", "Deviation %",
               "Detected by"]

    def __init__(self, data: pd.DataFrame, detector: AnomalyDetector):
        self.data = data
        self.detector = detector
        self.flags = {name: detector.flags(data[name]) for name in data.columns}
        rows = [row for name in data.columns for row in self._anomalies_of(name)]
        self.anomalies = pd.DataFrame(rows, columns=self.COLUMNS).sort_values(["Start", "Series"], ignore_index=True)
        self.anomalies.index += 1

    def statistics(self, names: list[str]) -> pd.DataFrame:
        data = self.data[names]
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        table = pd.DataFrame({"Mean": data.mean(), "Std dev": data.std(), "Median": data.median(), "IQR": q3 - q1})
        days = pd.DataFrame({name: self.flags[name][list(AnomalyDetector.TESTS)].sum() for name in names}).T
        anomalies = self.anomalies["Series"].value_counts().reindex(names, fill_value=0).rename("Anomalies")
        return table.join(days.astype(int)).join(anomalies)

    def anomalies_of(self, names: list[str]) -> pd.DataFrame:
        return self.anomalies[self.anomalies["Series"].isin(names)]

    @staticmethod
    def describe(anomaly: pd.Series) -> str:
        verb = "rose" if anomaly["Type"] == "Spike" else "fell"
        return (f"Output of {anomaly['Series']} {verb} to {anomaly['Value']:,.1f} on {anomaly['Peak day']:%d %b %Y}, "
                f"{anomaly['Deviation %']:+.1f}% against the expected {anomaly['Expected']:,.1f}. "
                f"The anomaly lasted {anomaly['Days']} day(s) and was found by: {anomaly['Detected by']}.")

    def _anomalies_of(self, name: str) -> list[dict]:
        flagged = self.flags[name].assign(Value=self.data[name]).query("Anomaly")
        gap = flagged.index.to_series().diff() != pd.Timedelta(days=1)
        new_run = gap | (flagged["Type"] != flagged["Type"].shift())
        return [self._summary(name, days) for _, days in flagged.groupby(new_run.cumsum())]

    @staticmethod
    def _summary(name: str, days: pd.DataFrame) -> dict:
        peak = (days["Value"] - days["Expected"]).abs().idxmax()
        value, expected = days.at[peak, "Value"], days.at[peak, "Expected"]
        tests = ", ".join(test for test in AnomalyDetector.TESTS if days[test].any())
        values = [name, days["Type"].iloc[0], days.index[0], days.index[-1], len(days), peak, value, expected,
                  (value / expected - 1) * 100 if expected else 0.0, tests]
        return dict(zip(Analysis.COLUMNS, values, strict=True))
