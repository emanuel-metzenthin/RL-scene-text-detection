import plotly.express as px
import pandas as pd

df = pd.DataFrame({'rec': [0.28, 0.27,0.26, 0.23, 0.09, 0.09], 'prec': [0.27, 0.27, 0.29, 0.30, 0.32, 0.33]})

fig = px.line(df,
              x='rec',
              y='prec',
              markers=True,
              symbol_sequence=['square'],
              labels={"0": "training iteration", "type": "metric"})

fig = fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01),
        font_size=30)

fig.show()
