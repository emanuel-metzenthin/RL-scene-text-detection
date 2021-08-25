import plotly.express as px
import pandas as pd

results = pd.read_csv("./experiments.csv")
score_cols = results.columns.drop(["experiment", "model", "dataset"])
results = results[results["experiment"] == "framestacking"].melt(id_vars=["model"], value_vars=score_cols, var_name="value", value_name="score")
results["metric"] = results.apply(
        lambda row: row["value"][:4] if row["value"] != "avg_iou" else "none",
        axis=1
)
results["value"] = results.apply(
        lambda row: row["value"].split("_")[1] if row["value"] != "avg_iou" else row["value"],
        axis=1
)
results = results[results["value"] != "avg_iou"]

px.bar(results,
       x="value",
       y="score",
       orientation="v",
       color="model",
       barmode="group",
       facet_row="metric",
       color_discrete_sequence=px.colors.qualitative.T10).show()
