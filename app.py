import pandas as pd
import streamlit as st

from mining.analysis import Analysis
from mining.anomaly_detector import AnomalyDetector
from mining.charts import Charts
from mining.data_source import DataSource
from mining.report import Report

SHEET_LINK = "https://docs.google.com/spreadsheets/d/1H3KIvVaE2NnfHb1dbUmVgFfeGQnNmxmq6to_4RAnpoY/edit"


class Dashboard:
    DATE = st.column_config.DateColumn(format="D MMM YYYY")
    NUMBER = st.column_config.NumberColumn(format="%.1f")
    ANOMALY_COLUMNS = {"Start": DATE, "End": DATE, "Peak day": DATE, "Value": NUMBER, "Expected": NUMBER,
                       "Deviation %": st.column_config.NumberColumn(format="%+.1f%%")}

    def render(self) -> None:
        link = self._source()
        data = self._load(link)
        names = st.sidebar.multiselect("Series", list(data.columns), default=list(data.columns))
        kind, degree = self._chart_settings()
        analysis = Analysis(data, self._detector())
        if not names:
            st.info("Select at least one series in the sidebar.")
            return
        self._header(analysis, names, kind, degree, "Google Sheets" if link else "Sample data")
        self._tabs(analysis, names, kind, degree)

    @staticmethod
    def _source() -> str:
        st.sidebar.title("WEYLAND-YUTANI")
        st.sidebar.caption("Mining Ops Analytics")
        link = st.sidebar.text_input("Google Sheets link", SHEET_LINK, placeholder="Empty = sample data",
                                     help="The sheet must be shared as 'Anyone with the link'.")
        if st.sidebar.button("Refresh data", icon=":material/refresh:", width="stretch"):
            st.cache_data.clear()
        return link.strip()

    @staticmethod
    @st.cache_data(ttl=60, show_spinner="Loading data...")
    def _read(link: str) -> pd.DataFrame:
        return DataSource().load(link)

    def _load(self, link: str) -> pd.DataFrame:
        try:
            return self._read(link)
        except ValueError as error:
            st.error(str(error), icon=":material/error:")
            st.stop()

    @staticmethod
    def _chart_settings() -> tuple[str, int | None]:
        st.sidebar.subheader("Chart")
        kind = st.sidebar.segmented_control("Chart type", Charts.TYPES, default="Line", required=True)
        degree = st.sidebar.select_slider("Trend line (polynomial degree)", ["Off", 1, 2, 3, 4], value=2)
        return kind, None if degree == "Off" else degree

    @staticmethod
    def _detector() -> AnomalyDetector:
        st.sidebar.subheader("Anomaly tests")
        side, default = st.sidebar, AnomalyDetector()
        return AnomalyDetector(
            iqr_k=side.number_input("IQR rule: k × IQR", 0.5, 5.0, default.iqr_k, 0.1),
            z_threshold=side.number_input("Z-score: |z| above", 1.0, 6.0, default.z_threshold, 0.1),
            ma_window=side.number_input("Moving average: window (days)", 2, 60, default.ma_window),
            ma_percent=side.number_input("Moving average: distance above (%)", 1.0, 200.0, default.ma_percent, 1.0),
            grubbs_alpha=side.select_slider("Grubbs' test: significance α", [0.10, 0.05, 0.01], default.grubbs_alpha),
            adjusted=side.toggle("Remove weekly pattern and trend", True,
                                 help="IQR, z-score and Grubbs test each day's deviation from its expected value."))

    def _header(self, analysis: Analysis, names: list[str], kind: str, degree: int | None, source: str) -> None:
        title, button = st.columns([4, 1], vertical_alignment="bottom")
        dates = analysis.data.index
        title.title("Daily resource extraction")
        title.caption(f"{dates[0]:%d %b %Y} – {dates[-1]:%d %b %Y} · {len(dates)} days · {source}")
        button.download_button("PDF report", type="primary", icon=":material/picture_as_pdf:", width="stretch",
                               data=lambda: Report(analysis, names, kind, degree, source).build(),
                               file_name=f"mining-report-{dates[0]:%Y-%m-%d}.pdf", mime="application/pdf")
        self._key_figures(analysis, names)

    @staticmethod
    def _key_figures(analysis: Analysis, names: list[str]) -> None:
        total, anomalies = analysis.data["Total"], analysis.anomalies_of(names)
        spikes = (anomalies["Type"] == "Spike").sum()
        tiles = st.columns(4)
        tiles[0].metric("Total output", f"{total.sum():,.0f}", border=True)
        tiles[1].metric("Mean daily output", f"{total.mean():,.1f}", f"median {total.median():,.1f}",
                        delta_color="off", delta_arrow="off", border=True)
        tiles[2].metric("Std deviation", f"{total.std():,.1f}", border=True)
        tiles[3].metric("Anomalies", len(anomalies), f"{spikes} spikes · {len(anomalies) - spikes} drops",
                        delta_color="off", delta_arrow="off", border=True)

    def _tabs(self, analysis: Analysis, names: list[str], kind: str, degree: int | None) -> None:
        charts = Charts(analysis)
        output, statistics, anomalies = st.tabs(["Output", "Statistics", "Anomalies"])
        output.pyplot(charts.output(names, kind, degree))
        statistics.dataframe(analysis.statistics(names).style.format("{:,.1f}", subset=["Mean", "Std dev", "Median",
                                                                                          "IQR"]))
        with anomalies:
            self._anomalies(analysis, charts, names)

    def _anomalies(self, analysis: Analysis, charts: Charts, names: list[str]) -> None:
        table = analysis.anomalies_of(names)
        if table.empty:
            st.success("No anomalies with the current settings.")
            return
        rows = st.dataframe(table, on_select="rerun", selection_mode="single-row",
                            column_config=self.ANOMALY_COLUMNS).selection.rows
        anomaly = table.iloc[rows[0]] if rows else table.loc[table["Deviation %"].abs().idxmax()]
        st.subheader(f"{anomaly['Type']} at {anomaly['Series']}")
        st.write(Analysis.describe(anomaly), "Click a row to see another anomaly.")
        st.pyplot(charts.anomaly(anomaly))


st.set_page_config(page_title="Mining Ops Analytics", page_icon="", layout="wide")
Dashboard().render()
