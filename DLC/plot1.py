import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def main():
    output_dir = Path(os.environ.get("AUTOTUNE_OUTPUT_DIR", "/home/CI/autotune"))
    history_path = output_dir / "throughput_data_qwen3_8b.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"Qwen3 throughput history not found: {history_path}")

    history = pd.read_csv(history_path, parse_dates=["date"])
    required = {"date", "total_tokens_per_s"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(
            "Qwen3 throughput history is missing columns: "
            + ", ".join(sorted(missing))
        )

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=history["date"],
        y=history["total_tokens_per_s"],
        mode="lines+markers",
        name="Qwen3-8B total tokens/s",
        hovertemplate=(
            "<b>Date:</b> %{x|%Y-%m-%d}<br>"
            "<b>Total tokens/s:</b> %{y:.2f}<extra></extra>"
        ),
    ))
    figure.update_layout(
        title_text="Qwen3-8B Inference Throughput Tuning",
        title_x=0.5,
        height=600,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Total tokens/s",
    )

    output_path = Path(os.environ.get(
        "PLOT_OUTPUT_PATH",
        str(output_dir / "qwen3_8b_throughput.html"),
    ))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path))
    print(f"Generated Qwen3 throughput plot from {history_path}")
    print(f"Plot path: {output_path}")


if __name__ == "__main__":
    main()
