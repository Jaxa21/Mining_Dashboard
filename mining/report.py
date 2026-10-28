from datetime import datetime

from fpdf import FPDF
from fpdf.fonts import FontFace
from matplotlib.figure import Figure

from mining.analysis import Analysis
from mining.charts import Charts


class Report(FPDF):
    INK, MUTED, AMBER = (31, 35, 41), (107, 111, 118), (201, 133, 0)

    def __init__(self, analysis: Analysis, names: list[str], kind: str, degree: int | None, source: str):
        super().__init__(format="A4")
        self.analysis, self.charts, self.names = analysis, Charts(analysis), names
        self.kind, self.degree, self.source = kind, degree, source
        self.anomalies = analysis.anomalies_of(names)
        self.set_margins(16, 16)
        self.set_auto_page_break(True, margin=18)

    def build(self) -> bytes:
        self.add_page()
        self._title()
        self._summary()
        self._statistics()
        self._charts()
        self._anomalies()
        return bytes(self.output())

    def header(self) -> None:
        if self.page_no() > 1:
            self._style(8, self.MUTED)
            self.cell(0, 6, "WEYLAND-YUTANI CORPORATION  |  Mining Operations Report", new_x="LMARGIN", new_y="NEXT")
            self.ln(3)

    def footer(self) -> None:
        self.set_y(-12)
        self._style(7.5, self.MUTED)
        self.cell(0, 6, f"Generated {datetime.now():%d %b %Y, %H:%M}  |  Page {self.page_no()} of {{nb}}", align="R")

    def _title(self) -> None:
        dates = self.analysis.data.index
        self._style(9, self.AMBER, bold=True)
        self.cell(0, 6, "WEYLAND-YUTANI CORPORATION", new_x="LMARGIN", new_y="NEXT")
        self._style(22, self.INK, bold=True)
        self.cell(0, 12, "Mining Operations Report", new_x="LMARGIN", new_y="NEXT")
        self._text(f"{dates[0]:%d %b %Y} - {dates[-1]:%d %b %Y} ({len(dates)} days). Data source: {self.source}.")

    def _summary(self) -> None:
        total, spikes = self.analysis.data["Total"], (self.anomalies["Type"] == "Spike").sum()
        self._heading("1. Summary")
        self._text(f"Total output {total.sum():,.0f} units, {total.mean():,.1f} per day on average "
                   f"(median {total.median():,.1f}). {len(self.anomalies)} anomalies: {spikes} spikes and "
                   f"{len(self.anomalies) - spikes} drops.")
        self._table(["Test", "Setting"], self._settings(), col_widths=(1, 2))

    def _settings(self) -> list[list[str]]:
        detector = self.analysis.detector
        return [["IQR rule", f"outside Q1 - {detector.iqr_k} x IQR ... Q3 + {detector.iqr_k} x IQR"],
                ["Z-score", f"|z| > {detector.z_threshold}"],
                ["Moving average", f"> {detector.ma_percent:.0f}% from the {detector.ma_window}-day moving average"],
                ["Grubbs' test", f"iterative, two-sided, alpha = {detector.grubbs_alpha}"],
                ["Weekly pattern and trend", "removed before testing" if detector.adjusted else "kept"]]

    def _statistics(self) -> None:
        self._heading("2. Statistics")
        table = self.analysis.statistics(self.names)
        rows = [[name, *(f"{v:,.1f}" for v in row[:4]), *(f"{v:.0f}" for v in row[4:])]
                for name, row in table.iterrows()]
        self._table(["Series", *table.columns], rows)

    def _charts(self) -> None:
        trend = f", polynomial trend of degree {self.degree}" if self.degree else ""
        self._heading("3. Daily output")
        self._text(f"{self.kind} chart{trend}. Triangles and coloured bars mark anomalous days.")
        self._image(self.charts.output(self.names, self.kind, self.degree))
        for name in self.names:
            self._heading(name, level=2, space=70)
            self._image(self.charts.output([name], "Line", self.degree, height=3.0))

    def _anomalies(self) -> None:
        self.add_page()
        self._heading("4. Anomaly register")
        if self.anomalies.empty:
            self._text("No anomalies were found with the current settings.")
            return
        self._table(["#", "Series", "Type", "Period", "Days", "Peak", "Expected", "Dev.", "Detected by"],
                    self._register_rows(), col_widths=(4, 11, 7, 15, 5, 8, 9, 8, 30), font_size=7)
        self._anomaly_sections()

    def _register_rows(self) -> list[list[str]]:
        return [[str(n), a["Series"], a["Type"], f"{a['Start']:%d %b} - {a['End']:%d %b}", str(a["Days"]),
                 f"{a['Value']:,.1f}", f"{a['Expected']:,.1f}", f"{a['Deviation %']:+.1f}%", a["Detected by"]]
                for n, a in self.anomalies.iterrows()]

    def _anomaly_sections(self) -> None:
        for number, anomaly in self.anomalies.iterrows():
            self._heading(f"Anomaly #{number}: {anomaly['Type']} at {anomaly['Series']}", level=2, space=70)
            self._text(Analysis.describe(anomaly))
            self._image(self.charts.anomaly(anomaly))

    def _heading(self, text: str, level: int = 1, space: float = 30) -> None:
        if self.will_page_break(space):
            self.add_page()
        self.ln(2)
        self._style(14 if level == 1 else 11, self.INK, bold=True)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")

    def _text(self, text: str) -> None:
        self._style(9.5, self.INK)
        self.multi_cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def _table(self, header: list[str], rows: list[list[str]], font_size: float = 8, **options) -> None:
        self._style(font_size, self.INK)
        style = FontFace(emphasis="BOLD", color=255, fill_color=self.INK)
        with self.table(headings_style=style, line_height=5, borders_layout="HORIZONTAL_LINES",
                        text_align="LEFT", **options) as table:
            for values in [header, *rows]:
                table.row(values)
        self.ln(3)

    def _image(self, fig: Figure) -> None:
        if self.will_page_break(self.epw * fig.get_figheight() / fig.get_figwidth()):
            self.add_page()
        self.image(Charts.png(fig), w=self.epw)
        self.ln(2)

    def _style(self, size: float, color: tuple, bold: bool = False) -> None:
        self.set_font("helvetica", "B" if bold else "", size)
        self.set_text_color(*color)
