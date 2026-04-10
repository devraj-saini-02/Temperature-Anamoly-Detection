import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(df):
    results = {}
    gt = df['ground_truth'].astype(int)

    for method, col in [('Threshold-Only',    'detected_threshold'),
                         ('Threshold + Spike', 'detected_combined')]:
        pred = df[col].astype(int)
        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        tn = int(((pred == 0) & (gt == 0)).sum())

        prec = round(precision_score(gt, pred, zero_division=0), 4)
        rec  = round(recall_score(gt, pred, zero_division=0), 4)
        f1   = round(f1_score(gt, pred, zero_division=0), 4)

        results[method] = {
            'True Positives':  tp,
            'False Positives': fp,
            'True Negatives':  tn,
            'False Negatives': fn,
            'Precision':       prec,
            'Recall':          rec,
            'F1 Score':        f1
        }

    table1 = pd.DataFrame(results).T
    print("\n=== TABLE 1: Detection Performance ===")
    print(table1.to_string())

    # ── Scenario labelling ──────────────────────────────────────────────
    T1 = df.attrs.get('T1', df['temp'].mean())

    def label_scenario(row):
        if row['ground_truth']:
            return 'Failure Window'
        elif row['temp'] >= T1:
            return 'Pre-failure Slow Rise'
        elif row['instant_spike']:
            return 'Spike Episode'
        else:
            return 'Nominal'

    df = df.copy()
    df['scenario'] = df.apply(label_scenario, axis=1)

    rows = []
    for scenario in ['Nominal', 'Pre-failure Slow Rise', 'Spike Episode', 'Failure Window']:
        sub = df[df['scenario'] == scenario]
        if len(sub) == 0:
            continue
        for method, col in [('Threshold-Only',  'detected_threshold'),
                              ('Threshold+Spike', 'detected_combined')]:
            flagged = sub[sub[col]]
            pct      = round(sub[col].mean() * 100, 1)
            avg_temp = round(flagged['temp'].mean(), 2) if len(flagged) > 0 else 0.0
            rows.append({
                'Scenario':                   scenario,
                'Method':                     method,
                '% Time Flagged':             pct,
                'Avg Temp When Flagged (°C)': avg_temp
            })

    table2 = pd.DataFrame(rows)
    print("\n=== TABLE 2: Scenario-wise Behavior ===")
    print(table2.to_string(index=False))

    return table1, table2, df