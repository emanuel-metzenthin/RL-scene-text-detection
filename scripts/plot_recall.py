import plotly.express as px
import pandas as pd

supervised = pd.read_csv("../ic13_supervised.csv", index_col=0)
finetuned = pd.read_csv("../ic13_finetuned.csv", index_col=0)

supervised["model"] = "supervised"
finetuned["model"] = "finetuned"

supervised2 = supervised.copy()
supervised2 = supervised2.drop(columns=["precision"])
supervised = supervised.drop(columns=["recall"])
supervised["type"] = "precision"
supervised2["type"] = "recall"
supervised.columns = ["value", "model", "type"]
supervised2.columns = ["value", "model", "type"]

finetuned2 = finetuned.copy()
finetuned2 = finetuned2.drop(columns=["precision"])
finetuned = finetuned.drop(columns=["recall"])
finetuned["type"] = "precision"
finetuned2["type"] = "recall"
finetuned.columns = ["value", "model", "type"]
finetuned2.columns = ["value", "model", "type"]


df = pd.concat([supervised2, supervised, finetuned2, finetuned], axis=0)
df.index = df.index * 1000
fig = px.line(df,
              x=df.index,
              y="value",
              color="model",
              line_dash="type",
              labels={"0": "training iteration", "type": "metric", "model": "", "metric": ""})

fig = fig.update_layout(
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
        title=''),
        font_size=30,)

fig.show()
