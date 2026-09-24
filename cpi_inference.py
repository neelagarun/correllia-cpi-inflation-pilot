"""
Inference-only helpers for the trained CPI LSTM model.

Loads the saved checkpoint + the FRED-MD dataset and produces a forward
CPI forecast without retraining. Reuses the exact preprocessing/model
code from cpi_lstm_v4.py so inference always matches training.
"""

import pandas as pd
import torch

from cpi_lstm_v4 import CSV_PATH, MODEL_PATH, TARGET, DEVICE, load_and_transform, LSTMForecaster


def load_model(model_path=MODEL_PATH):
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = LSTMForecaster(
        input_size=ckpt["input_size"],
        n_horizons=len(ckpt["horizons"]),
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def latest_forecast(csv_path=CSV_PATH, model_path=MODEL_PATH):
    """
    runs the trained model on the most recent seq_len months of data
    and returns a forecast for each horizon the model was trained on.

    csv_path may be a file path or any file-like object (e.g. a Streamlit
    upload) as long as it's formatted like the FRED-MD training data --
    a 'sasdate' column, a target column, and the feature columns the
    model was trained on.

    Returns
    -------
    dict with:
      as_of_date  : last date in the dataset
      as_of_level : CPIAUCSL level at as_of_date
      forecasts   : dataframe with horizon, target_date, pred_delta, pred_level, pred_mom_pct
      history     : dataframe of date/level for the full series (for charting)
    """

    model, ckpt = load_model(model_path)

    try:
        feat_diff, target_level, dates = load_and_transform(csv_path, TARGET)
    except KeyError as e:
        raise ValueError(
            f"CSV is missing a required column: {e}. It must include a 'sasdate' "
            f"column and a '{TARGET}' column, formatted like the FRED-MD training data."
        ) from e

    cols = ckpt["selected_cols"]
    seq_len = ckpt["seq_len"]
    horizons = ckpt["horizons"]
    f_mean, f_std = ckpt["f_mean"], ckpt["f_std"]
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    missing_cols = [c for c in cols if c not in feat_diff.columns]
    if missing_cols:
        raise ValueError(
            "CSV is missing columns this model was trained on: " + ", ".join(missing_cols)
        )

    feat = feat_diff[cols].values
    feat_std = (feat - f_mean) / f_std

    if len(feat_std) < seq_len:
        raise ValueError(
            f"Need at least {seq_len + 1} months of history ({seq_len} usable rows "
            f"after differencing), but this file only has {len(feat_std) + 1}."
        )

    window = feat_std[-seq_len:]
    x = torch.tensor(window[None, :, :], dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        pred_std = model(x).cpu().numpy()[0]

    pred_delta = pred_std * y_std + y_mean

    as_of_date = pd.Timestamp(dates.iloc[-1])
    as_of_level = float(target_level.iloc[-1])

    rows = []
    for h, delta in zip(horizons, pred_delta):
        target_date = as_of_date + pd.DateOffset(months=int(h))
        pred_level = as_of_level + float(delta)
        rows.append({
            "horizon": int(h),
            "target_date": target_date,
            "pred_delta": float(delta),
            "pred_level": pred_level,
            "pred_mom_pct": float(delta) / as_of_level * 100,
        })

    forecasts = pd.DataFrame(rows)

    history = pd.DataFrame({
        "date": pd.to_datetime(dates.values),
        "level": target_level.values,
    })

    return {
        "as_of_date": as_of_date,
        "as_of_level": as_of_level,
        "forecasts": forecasts,
        "history": history,
    }


if __name__ == "__main__":
    result = latest_forecast()
    print(f"as of {result['as_of_date'].date()}: CPIAUCSL = {result['as_of_level']:.3f}")
    print(result["forecasts"].to_string(index=False))
