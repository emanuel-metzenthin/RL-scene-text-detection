import plotly.express as px
import pandas as pd

EXPERIMENT = "framestacking"
EXCLUDE_METRICS = ["ic15"]

results = pd.read_csv("../experiments.csv")
score_cols = results.columns.drop(["experiment", "model", "dataset"])
results = results[results["experiment"] == EXPERIMENT].melt(id_vars=["model"], value_vars=score_cols, var_name="value", value_name="score")
results["metric"] = results.apply(
        lambda row: row["value"][:4] if row["value"] != "avg_iou" else "none",
        axis=1
)
results["value"] = results.apply(
        lambda row: row["value"].split("_")[1] if row["value"] != "avg_iou" else row["value"],
        axis=1
)
results = results[(results["value"] != "avg_iou") & (~results["metric"].isin(EXCLUDE_METRICS))]

fig = px.bar(results,
       x="value",
       y="score",
       orientation="v",
       color="model",
       barmode="group",
       facet_row="metric",
       labels={"value": "metric", "model": ""},
       color_discrete_sequence=px.colors.qualitative.T10)

fig = fig.update_layout(
        legend=dict(
            yanchor="top",
            y=-0.15,
            xanchor="left",
            x=0.01,
            orientation="h"),
        font_size=30)

fig.show()

results = results[results["metric"] == "tiou"]

fig = px.bar(results,
       x="value",
       y="score",
       orientation="v",
       color="model",
       barmode="group",
       labels={"value": "ICDAR2013 metric", "model": ""},
       color_discrete_sequence=px.colors.qualitative.T10)\

fig = fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.30,
            xanchor="left",
            x=0.01),
        font_size=30)

fig.show()
