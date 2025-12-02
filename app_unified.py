"""
Suicide Rate Correlation Analysis Dashboard
============================================
Exploring correlations between US suicide rates and:
- Firearm availability/ownership
- Technology and social media adoption

Data Sources:
- Annual Crude Suicide Rates (1990-2023)
- Firearm suicide/homicide dataset (1949-present)
- NTIA Technology adoption surveys
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Output, Input, State, ctx
import dash_bootstrap_components as dbc
from scipy import stats
import csv
import re
import os

LIGHT_BG = "#f8f9fa"
LIGHT_CARD = "#ffffff"
LIGHT_BORDER = "#dee2e6"
ACCENT_CYAN = "#0d6efd"
ACCENT_ORANGE = "#dc3545"
ACCENT_GREEN = "#198754"
ACCENT_PURPLE = "#6f42c1"
ACCENT_YELLOW = "#ffc107"
TEXT_PRIMARY = "#212529"
TEXT_SECONDARY = "#6c757d"

DARK_BG = LIGHT_BG
DARK_CARD = LIGHT_CARD
DARK_BORDER = LIGHT_BORDER

REGION_COLORS = {
    'New England': '#0d6efd',
    'Mid-Atlantic': '#6f42c1',
    'Middle Atlantic': '#6f42c1',
    'East North Central': '#198754',
    'West North Central': '#fd7e14',
    'South Atlantic': '#e67700',
    'East South Central': '#dc3545',
    'West South Central': '#9c36b5',
    'Mountain': '#0dcaf0',
    'Pacific': '#20c997'
}

BAND_COLORS = {
    "< 10": "#44ce1b",    
    "10-15": "#bbdb44",   
    "15-20": "#f7e379",   
    "20-25": "#f2a134",   
    "25-30": "#e51f1f",   
    "> 30": "#8e1f1f"     
}
BAND_ORDER = ["< 10", "10-15", "15-20", "20-25", "25-30", "> 30"]


STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}

ABBREV_TO_STATE = {v: k for k, v in STATE_ABBREV.items()}

STATE_COORDS = {
    "Alabama": (-86.8, 32.8), "Alaska": (-153.5, 64.2), "Arizona": (-111.7, 34.3),
    "Arkansas": (-92.4, 34.9), "California": (-119.5, 37.2), "Colorado": (-105.5, 39.0),
    "Florida": (-81.7, 28.1), "Georgia": (-83.4, 32.6), "Hawaii": (-155.2, 20.5),
    "Idaho": (-114.5, 44.4), "Illinois": (-89.2, 40.0), "Indiana": (-86.3, 39.9),
    "Iowa": (-93.5, 42.0), "Kansas": (-98.4, 38.5), "Kentucky": (-85.3, 37.8),
    "Louisiana": (-91.9, 31.0), "Maine": (-69.2, 45.4), "Michigan": (-85.4, 44.3),
    "Minnesota": (-94.3, 46.3), "Mississippi": (-89.7, 32.7), "Missouri": (-92.5, 38.3),
    "Montana": (-109.6, 47.0), "Nebraska": (-99.8, 41.5), "Nevada": (-117.0, 39.5),
    "New Mexico": (-106.0, 34.4), "New York": (-75.5, 43.0), "North Carolina": (-79.4, 35.5),
    "North Dakota": (-100.5, 47.4), "Ohio": (-82.8, 40.3), "Oklahoma": (-97.5, 35.5),
    "Oregon": (-120.5, 44.0), "Pennsylvania": (-77.6, 40.9), "South Carolina": (-80.9, 33.9),
    "South Dakota": (-100.2, 44.4), "Tennessee": (-86.3, 35.8), "Texas": (-99.3, 31.5),
    "Utah": (-111.7, 39.3), "Virginia": (-78.8, 37.5), "Washington": (-120.5, 47.4),
    "West Virginia": (-80.6, 38.9), "Wisconsin": (-89.8, 44.6), "Wyoming": (-107.5, 43.0)
}

SMALL_STATES = {
    "Connecticut": {"origin": (-72.7, 41.6), "label": (-64.0, 40.2)},
    "Delaware": {"origin": (-75.5, 39.0), "label": (-64.0, 37.2)},
    "District of Columbia": {"origin": (-77.0, 38.9), "label": (-70.0, 35.5)},
    "Maryland": {"origin": (-76.6, 39.3), "label": (-64.0, 36.0)},
    "Massachusetts": {"origin": (-71.8, 42.2), "label": (-64.0, 42.6)},
    "New Hampshire": {"origin": (-71.5, 43.7), "label": (-64.0, 45.0)},
    "New Jersey": {"origin": (-74.4, 40.1), "label": (-64.0, 38.4)},
    "Rhode Island": {"origin": (-71.5, 41.7), "label": (-64.0, 41.4)},
    "Vermont": {"origin": (-72.6, 44.0), "label": (-64.0, 46.4)},
}

REGION_NAMES = {
    "NE": "New England", "MA": "Mid-Atlantic", "ENC": "East North Central",
    "WNC": "West North Central", "SA": "South Atlantic", "ESC": "East South Central",
    "WSC": "West South Central", "M": "Mountain", "P": "Pacific"
}


def load_suicide_rates():
    """Load and process suicide rates data."""
    df = pd.read_csv("Annual Crude Suicide Rates (rates per 100,000 population) in USA States, 1990-2023.csv")
    
   
    df["Region_Code"] = df["State / [Region]"].str.extract(r'\[([A-Z]+)\]')
    df["State"] = df["State / [Region]"].str.replace(r"\s*\[.*\]", "", regex=True)
    df = df.drop(columns=["State / [Region]"])
    

    df_states = df[df["State"] != "U.S.A."].copy()
    df_usa = df[df["State"] == "U.S.A."].copy()
    
   
    year_columns = [str(year) for year in range(1990, 2024)]
    
    df_long = df_states.melt(
        id_vars=["State", "Region_Code"], 
        value_vars=year_columns, 
        var_name="Year", 
        value_name="Rate"
    )
    df_long["Year"] = df_long["Year"].astype(int)
    df_long["Region"] = df_long["Region_Code"].map(REGION_NAMES)
    df_long["Abbrev"] = df_long["State"].map(STATE_ABBREV)
    
  
    df_long["Rate_Band"] = df_long["Rate"].apply(get_rate_band)
    
    
    df_usa_long = df_usa.melt(
        id_vars=["State"], 
        value_vars=year_columns, 
        var_name="Year", 
        value_name="Rate"
    )
    df_usa_long["Year"] = df_usa_long["Year"].astype(int)
    
    return df_long, df_usa_long


def get_rate_band(rate):
    """Categorize suicide rate into bands."""
    if rate < 10: return "< 10"
    elif rate < 15: return "10-15"
    elif rate < 20: return "15-20"
    elif rate < 25: return "20-25"
    elif rate < 30: return "25-30"
    else: return "> 30"


print("Loading suicide rates data...")
df_suicide, df_usa_suicide = load_suicide_rates()



df_suicide_focus = df_suicide[df_suicide["Year"] >= 2000].copy()
df_usa_focus = df_usa_suicide[df_usa_suicide["Year"] >= 2000].copy()

print(f"Loaded {len(df_suicide_focus)} state-year suicide records")


def calculate_correlation(x, y):
    """Calculate Pearson correlation with p-value."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5:
        return 0, 1
    return stats.pearsonr(x[mask], y[mask])



CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body {{
    background-color: {LIGHT_BG} !important;
    color: {TEXT_PRIMARY} !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}

.container-fluid {{
    background-color: {LIGHT_BG} !important;
}}

.card {{
    background-color: {LIGHT_CARD} !important;
    border: 1px solid {LIGHT_BORDER} !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}

.section-divider {{
    border-top: 1px solid {LIGHT_BORDER};
    margin: 80px 0;
}}

.hero-title {{
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, {ACCENT_CYAN}, {ACCENT_PURPLE});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    line-height: 1.2;
}}

.hero-subtitle {{
    font-size: 1.25rem;
    color: {TEXT_SECONDARY};
    font-weight: 400;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
}}

.section-number {{
    font-size: 0.875rem;
    font-weight: 600;
    color: {ACCENT_CYAN};
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
}}

.section-title {{
    font-size: 2rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin-bottom: 0.75rem;
}}

.section-description {{
    color: {TEXT_SECONDARY};
    font-size: 1.1rem;
    line-height: 1.6;
    max-width: 800px;
}}

.stat-card {{
    background: {LIGHT_CARD};
    border: 1px solid {LIGHT_BORDER};
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    min-height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}

.stat-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
}}

.stat-value {{
    font-size: 2.75rem;
    font-weight: 700;
    color: {ACCENT_CYAN};
    line-height: 1;
}}

.stat-label {{
    font-size: 0.9rem;
    color: {TEXT_SECONDARY};
    margin-top: 12px;
    line-height: 1.4;
}}

.insight-box {{
    background: linear-gradient(135deg, rgba(13, 110, 253, 0.06), rgba(111, 66, 193, 0.06));
    border: 1px solid rgba(13, 110, 253, 0.2);
    border-radius: 12px;
    padding: 24px;
    margin: 24px 0;
}}

.caveat-box {{
    background: rgba(220, 53, 69, 0.06);
    border: 1px solid rgba(220, 53, 69, 0.2);
    border-radius: 12px;
    padding: 24px;
    margin: 24px 0;
}}

.legend-item {{
    display: inline-block;
    padding: 8px 14px;
    margin: 4px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 13px;
    font-weight: 500;
}}

.legend-item:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}}

.narrative-text {{
    font-size: 1.05rem;
    line-height: 1.8;
    color: {TEXT_SECONDARY};
    max-width: 700px;
}}

.annotation-callout {{
    background: {DARK_CARD};
    border: 1px solid {DARK_BORDER};
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.875rem;
}}

.dropdown-dark .Select-control {{
    background-color: {DARK_CARD} !important;
    border-color: {DARK_BORDER} !important;
}}

.dropdown-dark .Select-menu-outer {{
    background-color: {DARK_CARD} !important;
    border-color: {DARK_BORDER} !important;
}}
"""


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>Beyond the Screen: Suicide Rate Correlations in America</title>
        {{%favicon%}}
        {{%css%}}
        <style>{CUSTOM_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
    
   
    html.Div([
        
        
        dbc.Row([
            dbc.Col([
                dbc.Button("▶ Play", id="play-button", color="info", className="me-2", size="sm"),
            ], width="auto"),
            dbc.Col([
                html.Label("Animation Speed:", className="me-2 text-secondary small"),
                dcc.Slider(
                    id="speed-slider", min=200, max=1500, step=100, value=700,
                    marks={200: "Fast", 700: "Med", 1500: "Slow"}, 
                    className="mt-2"
                )
            ], width=2),
            dbc.Col([
                html.Label("Year:", className="me-2 text-secondary small"),
                dcc.Slider(
                    id="year-slider", min=2000, max=2023, step=1, value=2023,
                    marks={y: str(y) for y in range(2000, 2024, 4)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], width=7),
        ], className="mb-3 align-items-center"),
        
       
        dbc.Row([
            dbc.Col([
                html.Label("Filter by rate (click to toggle):", className="me-3 text-secondary small"),
                html.Div([
                    html.Div(
                        id=f"legend-{band}", 
                        className="legend-item",
                        style={
                            "backgroundColor": BAND_COLORS[band],
                            "color": "#000" if band in ["< 10", "10-15", "15-20", "20-25"] else "#fff"
                        },
                        children=band,
                        n_clicks=0
                    ) for band in BAND_ORDER
                ], style={"display": "inline-flex", "flexWrap": "wrap"})
            ], className="d-flex align-items-center")
        ], className="mb-4"),
        
        dcc.Store(id="active-bands", data=BAND_ORDER),
        
       
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="choropleth-map", style={"height": "550px"}, 
                         config={"displayModeBar": True, "scrollZoom": True})
            ], md=8),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("State Details", className="mb-3", style={"color": TEXT_PRIMARY}),
                        html.Div(id="state-stats", className="mb-3"),
                        html.Hr(style={"borderColor": DARK_BORDER}),
                        html.Div(id="ranking-text", className="mb-2 text-secondary small"),
                        dcc.Graph(id="state-trend-chart", style={"height": "180px"}, 
                                 config={"displayModeBar": False, "scrollZoom": False}),
                        html.Hr(style={"borderColor": DARK_BORDER}),
                        dcc.Graph(id="regional-chart", style={"height": "200px"},
                                 config={"displayModeBar": False, "scrollZoom": False})
                    ])
                ], style={"backgroundColor": DARK_CARD, "border": f"1px solid {DARK_BORDER}"})
            ], md=4)
        ]),
    ]),
    
   
    dcc.Store(id="selected-state", data=None),
    dcc.Interval(id="play-interval", interval=700, disabled=True)
    
], fluid=True, style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px 40px"})



@callback(
    Output("active-bands", "data"),
    [Input(f"legend-{band}", "n_clicks") for band in BAND_ORDER],
    State("active-bands", "data"),
    prevent_initial_call=True
)
def toggle_band(*args):
    clicks = args[:-1]
    current_bands = args[-1]
    
    triggered = ctx.triggered_id
    if triggered:
        band = triggered.replace("legend-", "")
        if band in current_bands:
            if len(current_bands) > 1:
                current_bands = [b for b in current_bands if b != band]
        else:
            current_bands = current_bands + [band]
    
    return current_bands



@callback(
    [Output(f"legend-{band}", "style") for band in BAND_ORDER],
    Input("active-bands", "data")
)
def update_legend_styles(active_bands):
    styles = []
    for band in BAND_ORDER:
        is_active = band in active_bands
        
        text_color = "#212529" if not is_active else ("#000" if band in ["< 10", "10-15", "15-20", "20-25"] else "#fff")
        style = {
            "display": "inline-block",
            "padding": "8px 14px",
            "margin": "4px",
            "borderRadius": "8px",
            "cursor": "pointer",
            "fontSize": "13px",
            "fontWeight": "500",
            "backgroundColor": BAND_COLORS[band] if is_active else "#dee2e6",
            "color": text_color,
            "opacity": 1 if is_active else 0.4,
            "transition": "all 0.2s ease"
        }
        styles.append(style)
    return styles



@callback(
    Output("choropleth-map", "figure"),
    Input("year-slider", "value"),
    Input("active-bands", "data"),
    Input("selected-state", "data")
)
def update_map(year, active_bands, selected_state):
    df_year = df_suicide_focus[df_suicide_focus["Year"] == year].copy()
    
    fig = go.Figure()
    

    for band in BAND_ORDER:
        df_band = df_year[df_year["Rate_Band"] == band]
        is_active = band in active_bands
        
        fig.add_trace(go.Choropleth(
            locations=df_band["Abbrev"],
            z=[1] * len(df_band),
            locationmode="USA-states",
            colorscale=[[0, BAND_COLORS[band]], [1, BAND_COLORS[band]]],
            showscale=False,
            marker=dict(line=dict(color=DARK_BORDER, width=1)),
            hoverinfo="text",
            text=[f"{row['State']}<br>Rate: {row['Rate']:.1f} per 100k<br>Region: {row['Region']}" 
                  for _, row in df_band.iterrows()],
            visible=is_active,
            name=band
        ))
    

    df_inactive = df_year[~df_year["Rate_Band"].isin(active_bands)]
    if len(df_inactive) > 0:
        fig.add_trace(go.Choropleth(
            locations=df_inactive["Abbrev"],
            z=[1] * len(df_inactive),
            locationmode="USA-states",
            colorscale=[[0, "#e9ecef"], [1, "#e9ecef"]],
            showscale=False,
            marker=dict(line=dict(color=DARK_BORDER, width=0.5)),
            hoverinfo="text",
            text=[f"{row['State']} (filtered)" for _, row in df_inactive.iterrows()],
            name="Filtered"
        ))
    
 
    rate_lookup = dict(zip(df_year["State"], df_year["Rate"]))
    regular_states = [s for s in STATE_COORDS.keys() if s not in SMALL_STATES]
    
    for state in regular_states:
        if state in rate_lookup:
            rate = rate_lookup[state]
            lon, lat = STATE_COORDS[state]
            abbrev = STATE_ABBREV[state]
            text_color = "#212529" if rate < 25 else "#ffffff"
            
            fig.add_trace(go.Scattergeo(
                lon=[lon], lat=[lat], mode="text",
                text=[abbrev],
                textfont=dict(size=9, color=text_color, family="Inter"),
                showlegend=False, hoverinfo="skip"
            ))
    
   
    for state, coords in SMALL_STATES.items():
        if state in rate_lookup:
            abbrev = STATE_ABBREV[state]
            orig_lon, orig_lat = coords["origin"]
            label_lon, label_lat = coords["label"]
            
            fig.add_trace(go.Scattergeo(
                lon=[orig_lon, label_lon], lat=[orig_lat, label_lat],
                mode="lines", line=dict(color="#adb5bd", width=1),
                showlegend=False, hoverinfo="skip"
            ))
            
            fig.add_trace(go.Scattergeo(
                lon=[label_lon], lat=[label_lat], mode="text",
                text=[abbrev],
                textfont=dict(size=9, color=TEXT_PRIMARY, family="Inter"),
                showlegend=False, hoverinfo="skip"
            ))
    
   
    if selected_state and selected_state in STATE_ABBREV:
        fig.add_trace(go.Choropleth(
            locations=[STATE_ABBREV[selected_state]],
            z=[1],
            locationmode="USA-states",
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            showscale=False,
            marker=dict(line=dict(color=ACCENT_CYAN, width=3)),
            hoverinfo="skip"
        ))
    
    fig.update_layout(
        title=dict(text=f"Suicide Rates per 100,000 Population — {year}", x=0.5, 
                   font=dict(size=16, color=TEXT_PRIMARY)),
        geo=dict(
            scope="usa", showlakes=True, lakecolor=DARK_BG,
            bgcolor=DARK_BG, landcolor=DARK_CARD,
            projection_type="albers usa"
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG
    )
    
    return fig


@callback(
    Output("selected-state", "data"),
    Input("choropleth-map", "clickData"),
    #Input("firearm-scatter", "clickData"),
    #Input("tech-scatter", "clickData"),
    State("selected-state", "data"),
    prevent_initial_call=True
)
def handle_state_click(map_click, current_state):
    triggered = ctx.triggered_id
    
    click_data = None
    if triggered == "choropleth-map":
        click_data = map_click
    # elif triggered == "firearm-scatter":
    #     click_data = firearm_click
    # elif triggered == "tech-scatter":
    #     click_data = tech_click
    
    if click_data:
        point = click_data["points"][0]
        
        
        customdata = point.get("customdata")
        if customdata and len(customdata) > 0:
            return customdata[0]
        
       
        text = point.get("text", "")
        if "<br>" in str(text):
            state_name = str(text).split("<br>")[0]
            if state_name in STATE_ABBREV:
                return state_name
        
       
        loc = point.get("location")
        if loc and loc in ABBREV_TO_STATE:
            return ABBREV_TO_STATE[loc]
    
    return current_state



@callback(
    Output("state-stats", "children"),
    Input("selected-state", "data")
)
def update_state_stats(state_name):
    if not state_name:
        return html.Div(
            "Click a state on the map to see details", 
            style={"color": TEXT_SECONDARY, "fontStyle": "italic"}
        )
    
    df_state = df_suicide_focus[df_suicide_focus["State"] == state_name].sort_values("Year")
    
    if df_state.empty:
        return html.Div(f"No data for {state_name}", style={"color": TEXT_SECONDARY})
    
    min_rate = df_state["Rate"].min()
    max_rate = df_state["Rate"].max()
    avg_rate = df_state["Rate"].mean()
    min_year = df_state.loc[df_state["Rate"].idxmin(), "Year"]
    max_year = df_state.loc[df_state["Rate"].idxmax(), "Year"]
    
 
    df_state = df_state.copy()
    df_state["Change"] = df_state["Rate"].diff()
    inc = df_state.loc[df_state["Change"].idxmax()]
    dec = df_state.loc[df_state["Change"].idxmin()]
    
    return html.Div([
        html.H5(state_name, className="mb-3", style={"color": ACCENT_CYAN}),
        html.P([html.Strong("Minimum: ", style={"color": TEXT_SECONDARY}), 
                f"{min_rate:.1f} ({int(min_year)})"], style={"color": TEXT_PRIMARY, "margin": "4px 0"}),
        html.P([html.Strong("Maximum: ", style={"color": TEXT_SECONDARY}), 
                f"{max_rate:.1f} ({int(max_year)})"], style={"color": TEXT_PRIMARY, "margin": "4px 0"}),
        html.P([html.Strong("Average: ", style={"color": TEXT_SECONDARY}), 
                f"{avg_rate:.1f}"], style={"color": TEXT_PRIMARY, "margin": "4px 0"}),
        html.Hr(style={"borderColor": DARK_BORDER, "margin": "12px 0"}),
        html.P([html.Strong("Largest Increase: ", style={"color": ACCENT_GREEN}), 
                f"+{inc['Change']:.1f} ({int(inc['Year']-1)}→{int(inc['Year'])})"], 
               style={"color": TEXT_PRIMARY, "margin": "4px 0", "fontSize": "0.9rem"}),
        html.P([html.Strong("Largest Decrease: ", style={"color": ACCENT_ORANGE}), 
                f"{dec['Change']:.1f} ({int(dec['Year']-1)}→{int(dec['Year'])})"], 
               style={"color": TEXT_PRIMARY, "margin": "4px 0", "fontSize": "0.9rem"}),
    ])



@callback(
    Output("ranking-text", "children"),
    Input("selected-state", "data"),
    Input("year-slider", "value")
)
def update_ranking(state_name, year):
    if not state_name:
        return ""
    
    df_year = df_suicide_focus[df_suicide_focus["Year"] == year].copy()
    df_year["Rank"] = df_year["Rate"].rank(ascending=False).astype(int)
    rank = df_year[df_year["State"] == state_name]["Rank"].values
    
    if len(rank) > 0:
        return f"Rank in {year}: #{rank[0]} of {len(df_year)}"
    return ""



@callback(
    Output("state-trend-chart", "figure"),
    Input("selected-state", "data")
)
def update_state_trend(state_name):
    fig = go.Figure()
    
    if not state_name:
        fig.add_annotation(text="Select a state", xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False, 
                          font=dict(size=14, color=TEXT_SECONDARY))
        fig.update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor=DARK_CARD, plot_bgcolor=DARK_CARD
        )
        return fig
    
    df_state = df_suicide_focus[df_suicide_focus["State"] == state_name]
    

    fig.add_trace(go.Scatter(
        x=df_usa_focus["Year"], y=df_usa_focus["Rate"],
        mode="lines", name="US Average",
        line=dict(color=TEXT_SECONDARY, dash="dash", width=2)
    ))
    

    fig.add_trace(go.Scatter(
        x=df_state["Year"], y=df_state["Rate"],
        mode="lines+markers", name=state_name,
        line=dict(color=ACCENT_CYAN, width=3), marker=dict(size=4)
    ))
    
    fig.update_layout(
        xaxis=dict(showgrid=False, color=TEXT_SECONDARY, tickfont=dict(size=10), fixedrange=True),
        yaxis=dict(title="Rate", range=[0, 35], showgrid=True, 
                   gridcolor=DARK_BORDER, color=TEXT_SECONDARY, tickfont=dict(size=10), fixedrange=True),
        margin=dict(l=40, r=10, t=10, b=30),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
                   font=dict(size=9, color=TEXT_SECONDARY)),
        hovermode="x unified",
        dragmode=False,
        paper_bgcolor=DARK_CARD, plot_bgcolor=DARK_CARD
    )
    return fig



@callback(
    Output("regional-chart", "figure"),
    Input("selected-state", "data"),
    Input("year-slider", "value")
)
def update_regional_chart(state_name, year):
    fig = go.Figure()
    
    if not state_name:
        fig.add_annotation(text="Select a state", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=14, color=TEXT_SECONDARY))
        fig.update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor=DARK_CARD, plot_bgcolor=DARK_CARD
        )
        return fig
    
    state_data = df_suicide_focus[(df_suicide_focus["State"] == state_name) & 
                                   (df_suicide_focus["Year"] == year)]
    if state_data.empty:
        return fig
    
    region = state_data["Region"].values[0]
    df_region = df_suicide_focus[
        (df_suicide_focus["Region"] == region) & 
        (df_suicide_focus["Year"] == year)
    ].copy()
    df_region = df_region.sort_values("Rate", ascending=True)
    
    colors = [ACCENT_CYAN if s == state_name else "#adb5bd" for s in df_region["State"]]
    
    fig.add_trace(go.Bar(
        x=df_region["Rate"], y=df_region["State"],
        orientation="h", marker_color=colors,
        text=df_region["Rate"].round(1), textposition="inside",
        textfont=dict(color="white", size=12),
        hovertemplate="%{y}: %{x:.1f}<extra></extra>"
    ))
    
    max_rate = df_region["Rate"].max()
    fig.update_layout(
        title=dict(text=f"{region} ({year})", font=dict(size=12, color=TEXT_SECONDARY)),
        xaxis=dict(title="Rate", range=[0, max_rate * 1.15], color=TEXT_SECONDARY, 
                  showgrid=False, tickfont=dict(size=10), fixedrange=True),
        yaxis=dict(color=TEXT_SECONDARY, tickfont=dict(size=9), fixedrange=True),
        margin=dict(l=100, r=10, t=30, b=30),
        showlegend=False,
        dragmode=False,
        paper_bgcolor=DARK_CARD, plot_bgcolor=DARK_CARD
    )
    return fig



@callback(
    Output("play-button", "children"),
    Output("play-interval", "disabled"),
    Input("play-button", "n_clicks"),
    State("play-interval", "disabled")
)
def toggle_play(n_clicks, is_disabled):
    if n_clicks is None:
        return "▶ Play", True
    return ("⏸ Pause", False) if is_disabled else ("▶ Play", True)


@callback(
    Output("play-interval", "interval"), 
    Input("speed-slider", "value")
)
def update_speed(speed):
    return speed


@callback(
    Output("year-slider", "value"),
    Input("play-interval", "n_intervals"),
    State("year-slider", "value"),
    State("play-interval", "disabled")
)
def advance_year(n_intervals, current_year, is_disabled):
    if is_disabled or n_intervals is None:
        return current_year
    return 2000 if current_year >= 2023 else current_year + 1


if __name__ == "__main__":
    app.run(debug=True, port=8050)

