import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

MONITOR_DIR = Path("monitoring")
MONITOR_DIR.mkdir(exist_ok=True)

reference = pd.read_parquet(MONITOR_DIR / "reference_eval.parquet")
current = pd.read_parquet(MONITOR_DIR / "current_eval.parquet")

report = Report(metrics=[
    DataDriftPreset()
])

report.run(
    reference_data=reference,
    current_data=current
)

report.save_html(str(MONITOR_DIR / "evidently_report.html"))

print("Evidently report generated via Python")
