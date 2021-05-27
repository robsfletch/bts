import numpy as np

def top_k_rate(estimator, X, y):
    data = X.copy()
    outcome = y.to_frame()
    probs = estimator.predict_proba(data)
    outcome['EstProb'] = probs[:, 1]

    selections = outcome.sort_values(['Date', 'EstProb'], ascending=[True, False])
    selections['pick_order'] = selections.groupby(['Date']).cumcount() + 1
    selections = selections[(selections.pick_order <= 15) & (selections.EstProb > .5)]

    selections['dcg']  = selections['Win'] / np.log2(selections['pick_order'] + 1)

    return selections['dcg'].mean()
