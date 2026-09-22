import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def _read_history(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["date", "total_tokens_per_s"])

    history = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "total_tokens_per_s"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {', '.join(sorted(missing))}"
        )
    history["total_tokens_per_s"] = pd.to_numeric(
        history["total_tokens_per_s"], errors="coerce"
    )
    return history.dropna(subset=["date", "total_tokens_per_s"])


def main() -> None:
    model_id = os.environ.get("MODEL", "qwen3_8b")
    history_dir = Path(
        os.environ.get(
            "AUTOTUNE_HISTORY_DIR",
            "/mnt/jfs/ci-dingtalk/autotune/qwen3_8b",
        )
    )
    tuning = _read_history(history_dir / f"throughput_data_{model_id}.csv")
    baseline = _read_history(history_dir / f"{model_id}_baseline_history.csv")
    if tuning.empty:
        raise FileNotFoundError(
            f"Tuning throughput history not found: {history_dir}"
        )
    if baseline.empty:
        raise FileNotFoundError(
            f"Baseline throughput history not found: {history_dir}"
        )

    tuning = tuning.sort_values("date")
    baseline = baseline.sort_values("date")
    model_name = os.environ.get("VLLM_PROFILE_MODEL_NAME", model_id)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=baseline["date"],
            y=baseline["total_tokens_per_s"],
            mode="lines+markers",
            name=f"{model_name} Baseline",
            line={"color": "#C44E52", "dash": "dot"},
            marker={"symbol": "triangle-up", "color": "#C44E52"},
            hovertemplate=(
                "<b>Date:</b> %{x|%Y-%m-%d}<br>"
                "<b>Baseline total tokens/s:</b> %{y:.2f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=tuning["date"],
            y=tuning["total_tokens_per_s"],
            mode="lines+markers",
            name=f"{model_name} Tune",
            line={"color": "#2878B5"},
            marker={"size": 8},
            hovertemplate=(
                "<b>Date:</b> %{x|%Y-%m-%d}<br>"
                "<b>Tune total tokens/s:</b> %{y:.2f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title_text=f"{model_name} Inference Throughput Tuning",
        title_x=0.5,
        height=600,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Total tokens/s",
    )

    output_path = Path(
        os.environ.get(
            "PLOT_OUTPUT_PATH",
            str(history_dir / f"{model_id}_throughput.html"),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path))
    print(f"Generated throughput plot from {history_dir}")
    print(f"Plot path: {output_path}")


if __name__ == "__main__":
    main()
