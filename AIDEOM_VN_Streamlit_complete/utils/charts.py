from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEAL = '#2a9d8f'
CORAL = '#e76f51'
NAVY = '#1f3a5f'
GOLD = '#f4a261'
PURPLE = '#7b2cbf'
PALETTE = [TEAL, CORAL, NAVY, GOLD, PURPLE, '#457b9d', '#06b6d4', '#84a98c']


def apply_theme(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        template='plotly_white',
        colorway=PALETTE,
        margin=dict(l=25, r=25, t=55, b=35),
        font=dict(family='Inter, Segoe UI, sans-serif', size=13, color='#0f172a'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.95)'
    )
    fig.update_xaxes(showgrid=True, gridcolor='#e8eef3')
    fig.update_yaxes(showgrid=True, gridcolor='#e8eef3')
    return fig


def line(df: pd.DataFrame, x: str, y: str | list[str], title: str) -> go.Figure:
    fig = px.line(df, x=x, y=y, markers=True, title=title)
    return apply_theme(fig)


def bar(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, title=title, text_auto='.2s')
    return apply_theme(fig)


def heatmap(z, x, y, title: str) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale=[[0, '#edf8f7'], [0.5, TEAL], [1, NAVY]], colorbar=dict(title='Giá trị')))
    fig.update_layout(title=title)
    return apply_theme(fig, 470)


def radar(categories: list[str], series: dict[str, list[float]], title: str) -> go.Figure:
    fig = go.Figure()
    cats = categories + [categories[0]]
    for name, vals in series.items():
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats, fill='toself', name=name))
    fig.update_layout(title=title, polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    return apply_theme(fig, 520)


def parallel(df: pd.DataFrame, dimensions: list[str], color_col: str, title: str) -> go.Figure:
    fig = px.parallel_coordinates(df, dimensions=dimensions, color=color_col, title=title, color_continuous_scale=[TEAL, GOLD, CORAL])
    return apply_theme(fig, 520)
