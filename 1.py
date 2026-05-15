import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
import os
from retry_requests import retry

# --- CONFIGURATION ---
PREVIOUS_SEASON_RANKS = {
    'Man City': 1500, 'Arsenal': 1450, 'Liverpool': 1400,
    'Aston Villa': 1200, 'Tottenham': 1180, 'Chelsea': 1100,
    'Newcastle': 1100, 'Man United': 1050, 'West Ham': 1000,
    'Brighton': 1000, 'Bournemouth': 900, 'Everton': 850,
    'Nott\'m Forest': 850, 'Crystal Palace': 900, 'Fulham': 900,
    'Brentford': 850, 'Wolves': 850, 'Leicester': 800, 'Ipswich': 750, 'Southampton': 750
}

STADIUM_COORDS = {
    'Liverpool': (53.4308, -2.9608), 'Arsenal': (51.5549, -0.1084),
    'Man City': (53.4831, -2.2004), 'Aston Villa': (52.5091, -1.8848),
    'Newcastle': (54.9756, -1.6217), 'West Ham': (51.5383, -0.0166)
}

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


# --- HELPERS ---
def get_weather(team, date):
    if team not in STADIUM_COORDS: return 15.0, 0.0
    lat, lon = STADIUM_COORDS[team]
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": lat, "longitude": lon, "start_date": date, "end_date": date, "daily": "precipitation_sum"}
    try:
        res = openmeteo.weather_api(url, params=params)[0].Daily()
        return 15.0, res.Variables(0).ValuesAsNumpy()[0]
    except:
        return 15.0, 0.0


class BayesianPredictor:
    def __init__(self, c=0.45):
        self.c = c
        self.ranks = PREVIOUS_SEASON_RANKS.copy()

    def get_rank(self, team):
        return self.ranks.get(team, 1000.0)

    def predict(self, home, away, h_perf, rain):
        r_h, r_a = self.get_rank(home), self.get_rank(away)
        prior = r_h / (r_h + r_a)
        weather_factor = 0.8 if rain > 5 else 1.0
        nudge = (prior ** 2 / 1000) * h_perf * weather_factor
        return np.clip(prior + (nudge * 0.2), 0.05, 0.95)

    def update(self, home, away, outcome):
        r_h, r_a = self.get_rank(home), self.get_rank(away)
        adjust = (self.c * (r_h - r_a)) - r_h
        self.ranks[home] = max(100, r_h + (adjust * outcome))


# --- EXECUTION PIPELINE ---
def run_pipeline():
    # UPDATE THIS: Points directly to the .xlsx file
    filename = 'all-euro-data-2025-2026.xlsx'

    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return None

    try:
        # We specify the SHEET NAME here (E0 = Premier League)
        df = pd.read_excel(filename, sheet_name='E0')
        print("Excel data loaded successfully!")
    except Exception as e:
        print(f"Failed to load Excel: {e}")
        return None

    # Filter columns
    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'HS', 'HST', 'B365H']
    df = df[cols].copy()
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    # Feature Engineering
    df['h_eff'] = (df['HST'] / df['HS'].replace(0, 1)).clip(0, 1)
    df['home_rolling_perf'] = df.groupby('HomeTeam')['h_eff'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(1.0)

    model = BayesianPredictor()
    results = []
    correct_preds, total_bets, total_profit = 0, 0, 0

    print("Analyzing Multimodal Pipeline...")
    for _, row in df.iterrows():
        _, rain = get_weather(row['HomeTeam'], row['Date'])
        prob = model.predict(row['HomeTeam'], row['AwayTeam'], row['home_rolling_perf'], rain)
        market_prob = 1 / row['B365H'] if pd.notnull(row['B365H']) else 0.5

        has_value = prob > market_prob
        bet_pnl = (row['B365H'] - 1) if (has_value and row['FTR'] == 'H') else -1 if has_value else 0

        if (prob > 0.5 and row['FTR'] == 'H') or (prob <= 0.5 and row['FTR'] != 'H'):
            correct_preds += 1
        if has_value:
            total_bets += 1
            total_profit += bet_pnl

        results.append({
            'Match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
            'Model_P': round(prob, 3),
            'Market_P': round(market_prob, 3),
            'Value': "YES" if has_value else "NO",
            'Result': row['FTR']
        })
        model.update(row['HomeTeam'], row['AwayTeam'],
                     (1.0 if row['FTR'] == 'H' else 0.5 if row['FTR'] == 'D' else 0.0))

    accuracy = (correct_preds / len(df)) * 100
    roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0

    print(f"\n--- GROUP 11 FINAL SUMMARY ---")
    print(f"Accuracy: {accuracy:.2f}% | ROI: {roi:.2f}% | Bets: {total_bets}")
    return pd.DataFrame(results)


if __name__ == "__main__":
    final_df = run_pipeline()
    if final_df is not None:
        print(final_df.tail(10))