from io import BytesIO

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mining.analysis import Analysis


class Charts:
    TYPES = ("Line", "Bar", "Stacked")
    PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
               "#8c6d31", "#17a2b8"]
    TYPE_STYLE = {"Spike": ("^", "#0a8f0a"), "Drop": ("v", "#d03b3b")}
    PADDING = pd.Timedelta(days=7)

    def __init__(self, analysis: Analysis):
        self.analysis = analysis
        mines = [name for name in analysis.data.columns if name != "Total"]
        self.colors = {name: self.PALETTE[i % len(self.PALETTE)] for i, name in enumerate(mines)} | {"Total": "#2b2b2b"}

    def output(self, names: list[str], kind: str = "Line", degree: int | None = None, height: float = 4.2) -> Figure:
        fig, ax = self._figure(height)
        bars = [name for name in names if kind != "Line" and name != "Total"]
        self._bars(ax, bars, stacked=kind == "Stacked")
        for name in names:
            if name not in bars:
                self._line(ax, name, self.analysis.data[name])
            if degree and not (kind == "Stacked" and name in bars):
                self._trend(ax, name, degree)
        return self._finish(fig, ax)

    def anomaly(self, anomaly: pd.Series) -> Figure:
        fig, ax = self._figure(2.8)
        name, window = anomaly["Series"], slice(anomaly["Start"] - self.PADDING, anomaly["End"] + self.PADDING)
        ax.axvspan(anomaly["Start"] - pd.Timedelta(hours=12), anomaly["End"] + pd.Timedelta(hours=12),
                   color=self.TYPE_STYLE[anomaly["Type"]][1], alpha=0.12)
        ax.plot(self.analysis.flags[name]["Expected"][window], color="#9a9890", linestyle=":", label="Expected")
        self._line(ax, name, self.analysis.data[name][window])
        return self._finish(fig, ax)

    @staticmethod
    def png(fig: Figure) -> BytesIO:
        image = BytesIO()
        fig.savefig(image, format="png", dpi=150, bbox_inches="tight")
        image.seek(0)
        return image

    def _line(self, ax: Axes, name: str, values: pd.Series) -> None:
        ax.plot(values, color=self.colors[name], linewidth=2 if name == "Total" else 1.4, label=name)
        flags = self.analysis.flags[name].loc[values.index]
        for kind, (marker, color) in self.TYPE_STYLE.items():
            days = flags.index[flags["Anomaly"] & (flags["Type"] == kind)]
            ax.scatter(days, values[days], marker=marker, color=color, s=40, zorder=5, label=kind)

    def _bars(self, ax: Axes, names: list[str], stacked: bool) -> None:
        width, bottom = 0.8 / (1 if stacked else max(len(names), 1)), 0
        for i, name in enumerate(names):
            flags, values = self.analysis.flags[name], self.analysis.data[name]
            colors = np.where(flags["Anomaly"], flags["Type"].map(lambda t: self.TYPE_STYLE[t][1]), self.colors[name])
            offset = pd.Timedelta(days=0 if stacked else (i - (len(names) - 1) / 2) * width)
            ax.bar(values.index + offset, values, width, bottom=bottom, color=colors, label=name)
            bottom = bottom + values if stacked else 0

    def _trend(self, ax: Axes, name: str, degree: int) -> None:
        values = self.analysis.data[name]
        x = np.linspace(0, 1, len(values))
        fitted = np.polyval(np.polyfit(x, values, min(degree, len(values) - 1)), x)
        r_squared = 1 - ((values - fitted) ** 2).sum() / ((values - values.mean()) ** 2).sum()
        ax.plot(values.index, fitted, color=self.colors[name], linestyle="--", linewidth=1,
                label=f"{name} trend (R² {r_squared:.2f})")

    @staticmethod
    def _figure(height: float) -> tuple[Figure, Axes]:
        fig = Figure(figsize=(10, height))
        return fig, fig.subplots()

    @staticmethod
    def _finish(fig: Figure, ax: Axes) -> Figure:
        ax.grid(axis="y", color="#e4e4df", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=True))
        ax.legend(unique.values(), unique.keys(), loc="lower left", bbox_to_anchor=(0, 1), fontsize=7.5, frameon=False,
                  ncols=6)
        fig.tight_layout()
        return fig
