"""
Premier League Match Predictor
Builds rolling features from raw season CSVs (2000-2022) and trains a
3-class classifier: H (home win) / D (draw) / A (away win).

Usage:
    python pl_predictor.py                    # train + example predictions
    python pl_predictor.py --predict "Arsenal" "Chelsea"
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split

warnings.filterwarnings('ignore')

RAW_DIR = "/Users/arthurfok/Downloads/pl_data/Datasets"
SEASONS = sorted(glob.glob(os.path.join(RAW_DIR, "20*.csv")))
# Also include the two extra files at the top level
EXTRA = [
    "/Users/arthurfok/Downloads/pl_data/2020-2021.csv",
    "/Users/arthurfok/Downloads/pl_data/2021-2022.csv",
]

RESULT_PTS = {'W': 3, 'D': 1, 'L': 0}
WINDOW = 5  # rolling form window


# ── Feature engineering ────────────────────────────────────────────────────

def _result_for(home_goals: int, away_goals: int, perspective: str) -> str:
    """Return W/D/L from team's perspective ('H' or 'A')."""
    if home_goals == away_goals:
        return 'D'
    if home_goals > away_goals:
        return 'W' if perspective == 'H' else 'L'
    return 'L' if perspective == 'H' else 'W'


def engineer_season(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling features to a single season's raw match data."""
    needed = {'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].dropna().copy()
    df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
    df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
    df = df.dropna()
    df = df.reset_index(drop=True)

    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

    # Per-team running stats: pts, gd, last-N results
    stats = {t: {'pts': 0, 'gd': 0, 'results': [], 'gs': 0, 'gc': 0} for t in teams}
    rows = []

    for _, match in df.iterrows():
        ht, at = match['HomeTeam'], match['AwayTeam']
        hs, as_ = stats[ht], stats[at]

        def form_pts(results):
            return sum(RESULT_PTS.get(r, 0) for r in results[-WINDOW:])

        def streak(results, outcome, n):
            tail = results[-n:]
            return int(all(r == outcome for r in tail) and len(tail) == n)

        row = {
            'HomeTeam': ht, 'AwayTeam': at, 'FTR': match['FTR'],
            # season cumulative
            'HTP': hs['pts'], 'ATP': as_['pts'],
            'HTGD': hs['gd'], 'ATGD': as_['gd'],
            'HTGS': hs['gs'], 'ATGS': as_['gs'],
            'HTGC': hs['gc'], 'ATGC': as_['gc'],
            # form
            'HTFormPts': form_pts(hs['results']),
            'ATFormPts': form_pts(as_['results']),
            'DiffPts': hs['pts'] - as_['pts'],
            'DiffFormPts': form_pts(hs['results']) - form_pts(as_['results']),
            # streaks
            'HTWinStreak3': streak(hs['results'], 'W', 3),
            'HTWinStreak5': streak(hs['results'], 'W', 5),
            'HTLossStreak3': streak(hs['results'], 'L', 3),
            'HTLossStreak5': streak(hs['results'], 'L', 5),
            'ATWinStreak3': streak(as_['results'], 'W', 3),
            'ATWinStreak5': streak(as_['results'], 'W', 5),
            'ATLossStreak3': streak(as_['results'], 'L', 3),
            'ATLossStreak5': streak(as_['results'], 'L', 5),
        }
        # last-5 form encoded numerically
        for i in range(1, WINDOW + 1):
            h_res = hs['results'][-(i)] if len(hs['results']) >= i else 'D'
            a_res = as_['results'][-(i)] if len(as_['results']) >= i else 'D'
            row[f'HM{i}'] = RESULT_PTS.get(h_res, 1)
            row[f'AM{i}'] = RESULT_PTS.get(a_res, 1)

        # Derived ratio features (safe division)
        hg_played = max(len(hs['results']), 1)
        ag_played = max(len(as_['results']), 1)
        row['HGoalRate'] = hs['gs'] / hg_played
        row['AGoalRate'] = as_['gs'] / ag_played
        row['HConcedRate'] = hs['gc'] / hg_played
        row['AConcedRate'] = as_['gc'] / ag_played

        rows.append(row)

        # Update stats AFTER recording (avoid data leakage)
        h_result = _result_for(int(match['FTHG']), int(match['FTAG']), 'H')
        a_result = _result_for(int(match['FTHG']), int(match['FTAG']), 'A')
        hg, ag = int(match['FTHG']), int(match['FTAG'])

        hs['pts'] += RESULT_PTS[h_result]
        hs['gd'] += hg - ag
        hs['gs'] += hg
        hs['gc'] += ag
        hs['results'].append(h_result)

        as_['pts'] += RESULT_PTS[a_result]
        as_['gd'] += ag - hg
        as_['gs'] += ag
        as_['gc'] += hg
        as_['results'].append(a_result)

    return pd.DataFrame(rows)


def load_all_seasons() -> pd.DataFrame:
    frames = []
    for path in SEASONS + EXTRA:
        try:
            df = pd.read_csv(path, encoding='latin1', low_memory=False)
            engineered = engineer_season(df)
            if not engineered.empty:
                frames.append(engineered)
        except Exception as e:
            print(f"  Skipping {os.path.basename(path)}: {e}")
    combined = pd.concat(frames, ignore_index=True)
    # Drop first ~5 matches per team (no history yet)
    combined = combined[combined['HTP'] + combined['ATP'] > 0]
    return combined


# ── Feature columns ────────────────────────────────────────────────────────

FEATURE_COLS = [
    'HTP', 'ATP', 'HTGD', 'ATGD', 'HTGS', 'ATGS', 'HTGC', 'ATGC',
    'HTFormPts', 'ATFormPts', 'DiffPts', 'DiffFormPts',
    'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5',
    'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
    'HM1', 'HM2', 'HM3', 'HM4', 'HM5',
    'AM1', 'AM2', 'AM3', 'AM4', 'AM5',
    'HGoalRate', 'AGoalRate', 'HConcedRate', 'AConcedRate',
]


# ── Predictor class ────────────────────────────────────────────────────────

class PLPredictor:
    """Trained Premier League match outcome predictor."""

    def __init__(self):
        self.model = None
        self.team_stats: dict[str, dict] = {}
        self.teams: list[str] = []

    def fit(self, df: pd.DataFrame):
        X = df[FEATURE_COLS]
        y = df['FTR']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        candidates = {
            'RandomForest': RandomForestClassifier(
                n_estimators=500, max_depth=8, min_samples_leaf=5,
                random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            ),
            'LogisticRegression': LogisticRegression(max_iter=2000, C=1.0),
        }

        print("=== Training Models ===\n")
        best_acc, best_name, best_model = 0, '', None
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            cv = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
            print(f"  {name:<22} test={acc:.3f}  cv={cv:.3f}")
            if acc > best_acc:
                best_acc, best_name, best_model = acc, name, model

        print(f"\nBest: {best_name} ({best_acc:.3f})")
        print("\nClassification report:")
        print(classification_report(y_test, best_model.predict(X_test), target_names=['Away Win', 'Draw', 'Home Win']))

        self.model = best_model
        self._build_team_profiles(df)

    def _build_team_profiles(self, df: pd.DataFrame):
        """Store last-known stats per team from the most recent data."""
        all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        for team in all_teams:
            home_rows = df[df['HomeTeam'] == team].tail(10)
            away_rows = df[df['AwayTeam'] == team].tail(10)

            def last_home(col, alt_col=None):
                if len(home_rows):
                    return home_rows[col].iloc[-1]
                if alt_col and len(away_rows):
                    return away_rows[alt_col].iloc[-1]
                return 0

            def last_away(col, alt_col=None):
                if len(away_rows):
                    return away_rows[col].iloc[-1]
                if alt_col and len(home_rows):
                    return home_rows[alt_col].iloc[-1]
                return 0

            self.team_stats[team] = {
                'HTP': last_home('HTP'), 'ATP': last_away('ATP'),
                'HTGD': last_home('HTGD'), 'ATGD': last_away('ATGD'),
                'HTGS': last_home('HTGS'), 'ATGS': last_away('ATGS'),
                'HTGC': last_home('HTGC'), 'ATGC': last_away('ATGC'),
                'HTFormPts': last_home('HTFormPts'), 'ATFormPts': last_away('ATFormPts'),
                'HTWinStreak3': last_home('HTWinStreak3'), 'ATWinStreak3': last_away('ATWinStreak3'),
                'HTWinStreak5': last_home('HTWinStreak5'), 'ATWinStreak5': last_away('ATWinStreak5'),
                'HTLossStreak3': last_home('HTLossStreak3'), 'ATLossStreak3': last_away('ATLossStreak3'),
                'HTLossStreak5': last_home('HTLossStreak5'), 'ATLossStreak5': last_away('ATLossStreak5'),
                'HM1': 1, 'HM2': 1, 'HM3': 1, 'HM4': 1, 'HM5': 1,
                'AM1': 1, 'AM2': 1, 'AM3': 1, 'AM4': 1, 'AM5': 1,
                'HGoalRate': last_home('HGoalRate') if 'HGoalRate' in (home_rows.columns if len(home_rows) else []) else 1.3,
                'AGoalRate': last_away('AGoalRate') if 'AGoalRate' in (away_rows.columns if len(away_rows) else []) else 1.1,
                'HConcedRate': last_home('HConcedRate') if 'HConcedRate' in (home_rows.columns if len(home_rows) else []) else 1.1,
                'AConcedRate': last_away('AConcedRate') if 'AConcedRate' in (away_rows.columns if len(away_rows) else []) else 1.3,
            }

        self.teams = sorted(self.team_stats.keys())

    def predict(self, home_team: str, away_team: str) -> dict:
        for t in [home_team, away_team]:
            if t not in self.team_stats:
                close = [x for x in self.teams if home_team.lower() in x.lower() or away_team.lower() in x.lower()]
                raise ValueError(f"Unknown team '{t}'. Did you mean: {close or self.teams}")

        h = self.team_stats[home_team]
        a = self.team_stats[away_team]

        row = {
            'HTP': h['HTP'], 'ATP': a['ATP'],
            'HTGD': h['HTGD'], 'ATGD': a['ATGD'],
            'HTGS': h['HTGS'], 'ATGS': a['ATGS'],
            'HTGC': h['HTGC'], 'ATGC': a['ATGC'],
            'HTFormPts': h['HTFormPts'], 'ATFormPts': a['ATFormPts'],
            'DiffPts': h['HTP'] - a['ATP'],
            'DiffFormPts': h['HTFormPts'] - a['ATFormPts'],
            'HTWinStreak3': h['HTWinStreak3'], 'HTWinStreak5': h['HTWinStreak5'],
            'HTLossStreak3': h['HTLossStreak3'], 'HTLossStreak5': h['HTLossStreak5'],
            'ATWinStreak3': a['ATWinStreak3'], 'ATWinStreak5': a['ATWinStreak5'],
            'ATLossStreak3': a['ATLossStreak3'], 'ATLossStreak5': a['ATLossStreak5'],
            'HM1': h['HM1'], 'HM2': h['HM2'], 'HM3': h['HM3'], 'HM4': h['HM4'], 'HM5': h['HM5'],
            'AM1': a['AM1'], 'AM2': a['AM2'], 'AM3': a['AM3'], 'AM4': a['AM4'], 'AM5': a['AM5'],
            'HGoalRate': h.get('HGoalRate', 1.3), 'AGoalRate': a.get('AGoalRate', 1.1),
            'HConcedRate': h.get('HConcedRate', 1.1), 'AConcedRate': a.get('AConcedRate', 1.3),
        }

        X = pd.DataFrame([row])[FEATURE_COLS]
        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_  # e.g. ['A', 'D', 'H']
        prob_map = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}
        prediction = classes[np.argmax(proba)]

        labels = {'H': f'{home_team} Win', 'D': 'Draw', 'A': f'{away_team} Win'}
        return {
            'home': home_team,
            'away': away_team,
            'prediction': labels[prediction],
            'probabilities': {
                labels['H']: prob_map.get('H', 0),
                labels['D']: prob_map.get('D', 0),
                labels['A']: prob_map.get('A', 0),
            }
        }

    def print_prediction(self, home: str, away: str):
        try:
            r = self.predict(home, away)
            print(f"\n{home} vs {away}")
            print(f"  Prediction → {r['prediction']}")
            for outcome, pct in r['probabilities'].items():
                bar = '█' * int(pct / 3)
                print(f"  {outcome:<25} {pct:>5.1f}%  {bar}")
        except ValueError as e:
            print(f"\n{home} vs {away}: {e}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', nargs=2, metavar=('HOME', 'AWAY'), help='Predict a single match')
    args = parser.parse_args()

    print("Loading and engineering features from all seasons...")
    df = load_all_seasons()
    print(f"Total matches: {len(df)}")
    print(f"Result distribution:\n{df['FTR'].value_counts().to_string()}\n")

    predictor = PLPredictor()
    predictor.fit(df)

    if args.predict:
        predictor.print_prediction(args.predict[0], args.predict[1])
    else:
        print("\n=== Example Predictions ===")
        fixtures = [
            ("Arsenal", "Chelsea"),
            ("Man United", "Liverpool"),
            ("Man City", "Tottenham"),
            ("Leicester", "Everton"),
            ("Newcastle", "West Ham"),
        ]
        for home, away in fixtures:
            predictor.print_prediction(home, away)

        print(f"\n\nAvailable teams ({len(predictor.teams)}):")
        for i, t in enumerate(predictor.teams):
            end = '\n' if (i + 1) % 5 == 0 else '  '
            print(f"  {t:<18}", end=end)
        print()

    return predictor


if __name__ == '__main__':
    predictor = main()
