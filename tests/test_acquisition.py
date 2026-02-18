import sys, os
import numpy as np
# ensure project root is on sys.path so tests can import app_al_demo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app_al_demo import make_demo_dataset, get_classifier, strategy_scores


def test_margin_scores_non_negative():
    X_train, X_test, y_train, y_test = make_demo_dataset(n_samples=80, n_features=8, n_classes=3, test_size=0.3, random_state=2)
    clf = get_classifier('LogisticRegression')
    clf.fit(X_train, y_train)
    scores = strategy_scores(clf, X_train[:10], 'margin_nodiv')
    assert np.all(scores >= 0), 'margin scores should be >= 0 (top1 - top2)'


def test_strategy_scores_length():
    X_train, X_test, y_train, y_test = make_demo_dataset(n_samples=80, n_features=8, n_classes=3, test_size=0.3, random_state=3)
    clf = get_classifier('LogisticRegression')
    clf.fit(X_train, y_train)
    for strat in ['least_confidence_nodiv', 'entropy_div', 'margin_div']:
        s = strategy_scores(clf, X_train[:5], strat)
        assert len(s) == 5
