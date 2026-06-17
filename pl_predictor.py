"""
Premier League Match Predictor — v2
Improvements over v1:
  - Head-to-head record between the two teams (last 5 meetings)
  - Home-specific and away-specific form (not blended)
  - Elo ratings (updated after every match)
  - Points-per-game rate (normalises early-season noise)
  - XGBoost + LightGBM in the model pool
  - Stacking ensemble: meta-learner on top of base predictions
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

RAW_DIR = "/Users/arthurfok/Downloads/pl_data/Datasets"
SEASONS = sorted(glob.glob(os.path.join(RAW_DIR, "20*.csv")))
EXTRA = [
    "/Users/arthurfok/Downloads/pl_data/2020-2021.csv",
    "/Users/arthurfok/Downloads/pl_data/2021-2022.csv",
]

RESULT_PTS = {'W': 3, 'D': 1, 'L': 0}
WINDOW = 5        # rolling form window
H2H_WINDOW = 5   # head-to-head history

# Elo constants
ELO_K = 32
ELO_BASE = 1500


# ── Helpers ───────────────────────────────────────────────────────────────

def _perspective(home_goals: int, away_goals: int, side: str) -> str:
    if home_goals == away_goals:
        return 'D'
    if home_goals > away_goals:
        return 'W' if side == 'H' else 'L'
    return 'L' if side == 'H' else 'W'


def elo_expected(ra: float, rb: float) -> float:
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def elo_update(ra: float, rb: float, score_a: float) -> tuple[float, float]:
    """score_a: 1=win, 0.5=draw, 0=loss for team A."""
    exp_a = elo_expected(ra, rb)
    new_ra = ra + ELO_K * (score_a - exp_a)
    new_rb = rb + ELO_K * ((1 - score_a) - (1 - exp_a))
    return new_ra, new_rb


# ── Feature engineering ───────────────────────────────────────────────────

def engineer_season(df: pd.DataFrame, team_elo: dict, h2h: dict) -> pd.DataFrame:
    needed = {'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].dropna().copy()
    df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
    df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
    df = df.dropna().reset_index(drop=True)

    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

    # Per-team running stats
    stats = {t: {
        'pts': 0, 'gd': 0, 'gs': 0, 'gc': 0,
        'results': [],         # all results W/D/L
        'home_results': [],    # results only as home team
        'away_results': [],    # results only as away team
        'played': 0,
    } for t in teams}

    # Ensure Elo entry for new teams
    for t in teams:
        if t not in team_elo:
            team_elo[t] = ELO_BASE

    rows = []

    for _, match in df.iterrows():
        ht, at = match['HomeTeam'], match['AwayTeam']
        hs, as_ = stats[ht], stats[at]
        hg, ag = int(match['FTHG']), int(match['FTAG'])

        def form_pts(results, n=WINDOW):
            return sum(RESULT_PTS.get(r, 0) for r in results[-n:])

        def streak(results, outcome, n):
            tail = results[-n:]
            return int(len(tail) == n and all(r == outcome for r in tail))

        def ppg(s):
            return s['pts'] / max(s['played'], 1)

        # Head-to-head
        pair = tuple(sorted([ht, at]))
        h2h_history = list(h2h.get(pair, []))[-H2H_WINDOW:]
        h2h_home_wins = sum(1 for r in h2h_history if r == ht)
        h2h_away_wins = sum(1 for r in h2h_history if r == at)
        h2h_draws     = sum(1 for r in h2h_history if r == 'D')

        row = {
            'HomeTeam': ht, 'AwayTeam': at, 'FTR': match['FTR'],

            # Season cumulative
            'HTP': hs['pts'], 'ATP': as_['pts'],
            'HTGD': hs['gd'], 'ATGD': as_['gd'],
            'HTGS': hs['gs'], 'ATGS': as_['gs'],
            'HTGC': hs['gc'], 'ATGC': as_['gc'],
            'HTPpg': ppg(hs), 'ATPpg': ppg(as_),

            # Overall form
            'HTFormPts': form_pts(hs['results']),
            'ATFormPts': form_pts(as_['results']),
            'DiffPts': hs['pts'] - as_['pts'],
            'DiffFormPts': form_pts(hs['results']) - form_pts(as_['results']),
            'DiffGD': hs['gd'] - as_['gd'],

            # Home-specific form for home team; away-specific for away team
            'HTHomeFormPts': form_pts(hs['home_results']),
            'ATAwayFormPts': form_pts(as_['away_results']),

            # Streaks
            'HTWinStreak3': streak(hs['results'], 'W', 3),
            'HTWinStreak5': streak(hs['results'], 'W', 5),
            'HTLossStreak3': streak(hs['results'], 'L', 3),
            'HTLossStreak5': streak(hs['results'], 'L', 5),
            'ATWinStreak3': streak(as_['results'], 'W', 3),
            'ATWinStreak5': streak(as_['results'], 'W', 5),
            'ATLossStreak3': streak(as_['results'], 'L', 3),
            'ATLossStreak5': streak(as_['results'], 'L', 5),

            # Goal rates
            'HGoalRate':   hs['gs'] / max(hs['played'], 1),
            'AGoalRate':   as_['gs'] / max(as_['played'], 1),
            'HConcedRate': hs['gc'] / max(hs['played'], 1),
            'AConcedRate': as_['gc'] / max(as_['played'], 1),

            # Last-5 results numerically
            **{f'HM{i}': RESULT_PTS.get(hs['results'][-(i)] if len(hs['results']) >= i else 'D', 1)
               for i in range(1, WINDOW + 1)},
            **{f'AM{i}': RESULT_PTS.get(as_['results'][-(i)] if len(as_['results']) >= i else 'D', 1)
               for i in range(1, WINDOW + 1)},

            # Elo
            'HElo': team_elo[ht],
            'AElo': team_elo[at],
            'DiffElo': team_elo[ht] - team_elo[at],
            'EloWinProb': elo_expected(team_elo[ht], team_elo[at]),

            # Head-to-head
            'H2HHomeWins': h2h_home_wins,
            'H2HAwayWins': h2h_away_wins,
            'H2HDraws':    h2h_draws,
            'H2HMatches':  len(h2h_history),
        }
        rows.append(row)

        # Update Elo
        score = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        team_elo[ht], team_elo[at] = elo_update(team_elo[ht], team_elo[at], score)

        # Update h2h
        winner = ht if hg > ag else (at if ag > hg else 'D')
        h2h.setdefault(pair, deque(maxlen=20)).append(winner)

        # Update season stats
        h_res = _perspective(hg, ag, 'H')
        a_res = _perspective(hg, ag, 'A')

        hs['pts'] += RESULT_PTS[h_res]
        hs['gd'] += hg - ag; hs['gs'] += hg; hs['gc'] += ag
        hs['results'].append(h_res); hs['home_results'].append(h_res); hs['played'] += 1

        as_['pts'] += RESULT_PTS[a_res]
        as_['gd'] += ag - hg; as_['gs'] += ag; as_['gc'] += hg
        as_['results'].append(a_res); as_['away_results'].append(a_res); as_['played'] += 1

    return pd.DataFrame(rows)


def load_all_seasons() -> pd.DataFrame:
    team_elo: dict[str, float] = {}   # persists across seasons
    h2h: dict[tuple, deque] = {}       # persists across seasons

    frames = []
    for path in SEASONS + EXTRA:
        try:
            df = pd.read_csv(path, encoding='latin1', low_memory=False)
            engineered = engineer_season(df, team_elo, h2h)
            if not engineered.empty:
                frames.append(engineered)
        except Exception as e:
            print(f"  Skipping {os.path.basename(path)}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined['HTP'] + combined['ATP'] > 0]
    return combined


# ── Feature columns ───────────────────────────────────────────────────────

FEATURE_COLS = [
    'HTP', 'ATP', 'HTGD', 'ATGD', 'HTGS', 'ATGS', 'HTGC', 'ATGC',
    'HTPpg', 'ATPpg',
    'HTFormPts', 'ATFormPts', 'DiffPts', 'DiffFormPts', 'DiffGD',
    'HTHomeFormPts', 'ATAwayFormPts',
    'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5',
    'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
    'HGoalRate', 'AGoalRate', 'HConcedRate', 'AConcedRate',
    'HM1', 'HM2', 'HM3', 'HM4', 'HM5',
    'AM1', 'AM2', 'AM3', 'AM4', 'AM5',
    'HElo', 'AElo', 'DiffElo', 'EloWinProb',
    'H2HHomeWins', 'H2HAwayWins', 'H2HDraws', 'H2HMatches',
]


# ── Predictor ─────────────────────────────────────────────────────────────

class PLPredictor:
    def __init__(self):
        self.model = None
        self.team_stats: dict[str, dict] = {}
        self.teams: list[str] = []

    def fit(self, df: pd.DataFrame):
        X = df[FEATURE_COLS]
        self.le = LabelEncoder()
        y = self.le.fit_transform(df['FTR'])  # A=0, D=1, H=2
        y_str = df['FTR']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        _, _, y_train_str, y_test_str = train_test_split(
            X, y_str, test_size=0.2, random_state=42, stratify=y
        )

        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=500, max_depth=10,
                                          min_samples_leaf=3, random_state=42, n_jobs=-1)),
            ('xgb', xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8,
                                       eval_metric='mlogloss', random_state=42,
                                       use_label_encoder=False)),
            ('lgbm', lgb.LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8,
                                         random_state=42, verbose=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                              learning_rate=0.05, subsample=0.8,
                                              random_state=42)),
        ]
        meta = LogisticRegression(max_iter=1000, C=1.0)
        stack = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta,
            cv=5,
            passthrough=False,
            n_jobs=-1,
        )

        print("=== Training Models ===\n")

        # Evaluate base models first
        results = {}
        for name, est in base_estimators:
            est.fit(X_train, y_train)
            acc = accuracy_score(y_test, est.predict(X_test))
            cv  = cross_val_score(est, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()
            print(f"  {name:<8} test={acc:.3f}  cv={cv:.3f}")
            results[name] = acc

        # Train stacking ensemble
        print("\n  Training stacking ensemble…")
        stack.fit(X_train, y_train)
        stack_acc = accuracy_score(y_test, stack.predict(X_test))
        stack_cv  = cross_val_score(stack, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()
        print(f"  {'stack':<8} test={stack_acc:.3f}  cv={stack_cv:.3f}")
        results['stack'] = stack_acc

        # Pick best
        named = {**dict(base_estimators), 'stack': stack}
        best_name = max(results, key=results.get)
        self.model = named[best_name]
        print(f"\nBest: {best_name} ({results[best_name]:.3f})")
        print("\nClassification report (best model):")
        preds = self.le.inverse_transform(self.model.predict(X_test))
        print(classification_report(y_test_str, preds,
                                     target_names=['Away Win', 'Draw', 'Home Win']))

        self._build_team_profiles(df)

    def _build_team_profiles(self, df: pd.DataFrame):
        all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        for team in all_teams:
            home_rows = df[df['HomeTeam'] == team].tail(10)
            away_rows = df[df['AwayTeam'] == team].tail(10)

            def lh(col, fallback=0):
                return home_rows[col].iloc[-1] if len(home_rows) else fallback

            def la(col, fallback=0):
                return away_rows[col].iloc[-1] if len(away_rows) else fallback

            self.team_stats[team] = {
                'HTP': lh('HTP'), 'ATP': la('ATP'),
                'HTGD': lh('HTGD'), 'ATGD': la('ATGD'),
                'HTGS': lh('HTGS'), 'ATGS': la('ATGS'),
                'HTGC': lh('HTGC'), 'ATGC': la('ATGC'),
                'HTPpg': lh('HTPpg', 1.5), 'ATPpg': la('ATPpg', 1.2),
                'HTFormPts': lh('HTFormPts'), 'ATFormPts': la('ATFormPts'),
                'HTHomeFormPts': lh('HTHomeFormPts'), 'ATAwayFormPts': la('ATAwayFormPts'),
                'HTWinStreak3': lh('HTWinStreak3'), 'HTWinStreak5': lh('HTWinStreak5'),
                'HTLossStreak3': lh('HTLossStreak3'), 'HTLossStreak5': lh('HTLossStreak5'),
                'ATWinStreak3': la('ATWinStreak3'), 'ATWinStreak5': la('ATWinStreak5'),
                'ATLossStreak3': la('ATLossStreak3'), 'ATLossStreak5': la('ATLossStreak5'),
                'HGoalRate': lh('HGoalRate', 1.3), 'AGoalRate': la('AGoalRate', 1.1),
                'HConcedRate': lh('HConcedRate', 1.1), 'AConcedRate': la('AConcedRate', 1.3),
                'HElo': lh('HElo', ELO_BASE), 'AElo': la('AElo', ELO_BASE),
                **{f'HM{i}': 1 for i in range(1, 6)},
                **{f'AM{i}': 1 for i in range(1, 6)},
            }

        self.teams = sorted(self.team_stats.keys())

    def predict(self, home_team: str, away_team: str) -> dict:
        for t in [home_team, away_team]:
            if t not in self.team_stats:
                close = [x for x in self.teams if t.lower() in x.lower()]
                raise ValueError(f"Unknown team '{t}'. Did you mean: {close or self.teams}")

        h = self.team_stats[home_team]
        a = self.team_stats[away_team]

        h_elo = h['HElo']
        a_elo = a['AElo']

        row = {
            'HTP': h['HTP'], 'ATP': a['ATP'],
            'HTGD': h['HTGD'], 'ATGD': a['ATGD'],
            'HTGS': h['HTGS'], 'ATGS': a['ATGS'],
            'HTGC': h['HTGC'], 'ATGC': a['ATGC'],
            'HTPpg': h['HTPpg'], 'ATPpg': a['ATPpg'],
            'HTFormPts': h['HTFormPts'], 'ATFormPts': a['ATFormPts'],
            'DiffPts': h['HTP'] - a['ATP'],
            'DiffFormPts': h['HTFormPts'] - a['ATFormPts'],
            'DiffGD': h['HTGD'] - a['ATGD'],
            'HTHomeFormPts': h['HTHomeFormPts'], 'ATAwayFormPts': a['ATAwayFormPts'],
            'HTWinStreak3': h['HTWinStreak3'], 'HTWinStreak5': h['HTWinStreak5'],
            'HTLossStreak3': h['HTLossStreak3'], 'HTLossStreak5': h['HTLossStreak5'],
            'ATWinStreak3': a['ATWinStreak3'], 'ATWinStreak5': a['ATWinStreak5'],
            'ATLossStreak3': a['ATLossStreak3'], 'ATLossStreak5': a['ATLossStreak5'],
            'HGoalRate': h['HGoalRate'], 'AGoalRate': a['AGoalRate'],
            'HConcedRate': h['HConcedRate'], 'AConcedRate': a['AConcedRate'],
            **{f'HM{i}': h[f'HM{i}'] for i in range(1, 6)},
            **{f'AM{i}': a[f'AM{i}'] for i in range(1, 6)},
            'HElo': h_elo, 'AElo': a_elo,
            'DiffElo': h_elo - a_elo,
            'EloWinProb': elo_expected(h_elo, a_elo),
            'H2HHomeWins': 0, 'H2HAwayWins': 0, 'H2HDraws': 0, 'H2HMatches': 0,
        }

        X = pd.DataFrame([row])[FEATURE_COLS]
        proba = self.model.predict_proba(X)[0]
        # Decode class indices back to A/D/H strings
        classes = self.le.inverse_transform(self.model.classes_)
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
            print(f"  → {r['prediction']}")
            for outcome, pct in r['probabilities'].items():
                print(f"  {outcome:<25} {pct:>5.1f}%  {'█' * int(pct / 3)}")
        except ValueError as e:
            print(f"\n{home} vs {away}: {e}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', nargs=2, metavar=('HOME', 'AWAY'))
    args = parser.parse_args()

    print("Loading and engineering features from all seasons...")
    df = load_all_seasons()
    print(f"Total matches: {len(df)}  |  Features: {len(FEATURE_COLS)}")
    print(f"Result distribution:\n{df['FTR'].value_counts().to_string()}\n")

    predictor = PLPredictor()
    predictor.fit(df)

    if args.predict:
        predictor.print_prediction(args.predict[0], args.predict[1])
    else:
        print("\n=== Example Predictions ===")
        for home, away in [("Arsenal", "Chelsea"), ("Man United", "Liverpool"),
                           ("Man City", "Tottenham"), ("Leicester", "Everton")]:
            predictor.print_prediction(home, away)

        print(f"\n\nAvailable teams ({len(predictor.teams)}): {', '.join(predictor.teams)}")

    return predictor


if __name__ == '__main__':
    predictor = main()
