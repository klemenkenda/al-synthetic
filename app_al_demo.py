import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
import random
import math


def make_demo_dataset(n_samples=200, n_features=32, n_classes=3, test_size=0.3, random_state=0):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=min(10, n_features),
                               n_redundant=0, n_classes=n_classes, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def load_npz_embeddings(path: Path):
    data = np.load(path)
    # Expect arrays named 'X' and 'y' or 'embeddings' and 'labels'
    if 'X' in data and 'y' in data:
        return data['X'], data['y']
    if 'embeddings' in data and 'labels' in data:
        return data['embeddings'], data['labels']
    raise RuntimeError('NPZ does not contain expected keys: use X/y or embeddings/labels')


def get_classifier(name: str):
    if name == 'LogisticRegression':
        return LogisticRegression(max_iter=1000)
    if name == 'RandomForest':
        return RandomForestClassifier(n_estimators=100)
    if name == 'SVC':
        return SVC(probability=True)
    if name == 'GradientBoosting':
        return GradientBoostingClassifier()
    raise ValueError('Unknown classifier')


def predict_proba_safe(clf, X):
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)
    # fallback: use decision_function and convert to softmax
    if hasattr(clf, 'decision_function'):
        scores = clf.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
    # final fallback: uniform
    n = X.shape[0]
    k = len(getattr(clf, 'classes_', [0]))
    return np.ones((n, k)) / max(k, 1)


def strategy_scores(clf, X_unlabeled, strategy: str):
    """Return acquisition scores for unlabeled points.

    Accepts strategy names with optional suffixes: e.g. 'least_confidence_div'
    or 'margin_nodiv'. The diversity suffix is ignored here (selection-time
    logic handles diversity); this function only returns uncertainty scores.
    """
    # allow 'least_confidence_div' / 'least_confidence_nodiv' etc.
    base = strategy.rsplit('_', 1)[0] if '_' in strategy else strategy
    probs = predict_proba_safe(clf, X_unlabeled)
    if base == 'random':
        return np.random.random(len(X_unlabeled))
    if base == 'least_confidence':
        return 1 - probs.max(axis=1)
    if base == 'margin':
        # margin between top-2 (top1 - top2); smaller margin == more uncertain
        top1 = probs.max(axis=1)
        top2 = np.partition(probs, -2, axis=1)[:, -2]
        return top1 - top2
    if base == 'entropy':
        ent = -np.sum(np.where(probs > 0, probs * np.log(probs), 0), axis=1)
        return ent
    raise ValueError(f'Unknown base strategy: {base}')


def run_random_baseline(X_pool, y_pool, X_test, y_test, classifier_name, seed_size, query_size, rounds, n_repeats=10):
    curves = []
    for rep in range(n_repeats):
        rng = np.random.RandomState(rep)
        unl = list(range(X_pool.shape[0]))
        # Ensure at least two classes in labeled set
        unique_classes = np.unique(y_pool)
        if seed_size >= 2 and len(unique_classes) >= 2:
            labeled = []
            # Pick one sample from each of the first two classes
            for cls in unique_classes[:2]:
                idx = rng.choice(np.where(y_pool == cls)[0], 1)[0]
                labeled.append(idx)
            # Fill the rest randomly
            remaining = [i for i in unl if i not in labeled]
            if seed_size > 2:
                labeled += rng.choice(remaining, size=seed_size - 2, replace=False).tolist()
            for i in labeled:
                unl.remove(i)
        else:
            # fallback: random sample
            labeled = rng.choice(unl, size=seed_size, replace=False).tolist()
            for i in labeled:
                unl.remove(i)
        # Check if at least two classes are present
        if len(np.unique(y_pool[labeled])) < 2:
            continue  # skip this repeat
        scores = []
        clf = get_classifier(classifier_name)
        for r in range(rounds + 1):
            # train
            clf.fit(X_pool[labeled], y_pool[labeled])
            preds = clf.predict(X_test)
            scores.append(f1_score(y_test, preds, average='macro'))
            if r == rounds:
                break
            chosen = rng.choice(unl, size=query_size, replace=False).tolist()
            for c in chosen:
                labeled.append(c)
                unl.remove(c)
        curves.append(scores)
    if len(curves) == 0:
        raise ValueError('Could not initialize labeled set with at least two classes in any repeat.')
    mean_curve = np.mean(curves, axis=0)
    return mean_curve


def main():
    st.title('Interactive Active Learning Demo')
    st.write('Run an interactive AL loop and compare to random sampling baseline.')

    # Sidebar controls
    st.sidebar.header('Settings')
    data_source = 'Synthetic'
    classifier_name = st.sidebar.selectbox('Classifier', ['LogisticRegression', 'RandomForest', 'SVC', 'GradientBoosting'])
    strategy = st.sidebar.selectbox(
        'Acquisition strategy',
        [
            'least_confidence_div', 'least_confidence_nodiv',
            'margin_div', 'margin_nodiv',
            'entropy_div', 'entropy_nodiv'
        ]
    )
    seed_size = st.sidebar.number_input('Seed labeled size', min_value=2, value=2, step=1)
    query_size = st.sidebar.number_input('Query size per round', min_value=1, value=1, step=1)
    rounds = st.sidebar.number_input('Rounds', min_value=1, value=100, step=1)


    if 'state' not in st.session_state:
        st.session_state.state = {}

    if st.button('Initialize'):
        # load or create dataset
        X_train, X_test, y_train, y_test = make_demo_dataset(n_samples=400, n_features=64, n_classes=3, test_size=0.3, random_state=1)

        # initialize labeled/unlabeled pools (indices)

        pool_size = X_train.shape[0]
        unlabeled = list(range(pool_size))
        # Ensure at least two classes in labeled set if possible
        unique_classes = np.unique(y_train)
        if seed_size >= 2 and len(unique_classes) >= 2:
            labeled = []
            # Pick one sample from each of the first two classes
            for cls in unique_classes[:2]:
                idx = np.where(y_train == cls)[0][0]
                labeled.append(idx)
            # Fill the rest randomly
            remaining = [i for i in unlabeled if i not in labeled]
            if seed_size > 2:
                labeled += random.sample(remaining, seed_size - 2)
            for idx in labeled:
                unlabeled.remove(idx)
        else:
            # fallback: random sample
            labeled = random.sample(unlabeled, seed_size)
            for idx in labeled:
                unlabeled.remove(idx)

        # store in session
        st.session_state.state.update({
            'X_pool': X_train,
            'y_pool': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'labeled': labeled,
            'unlabeled': unlabeled,
            'history': [],
            'classifier_name': classifier_name,
            'strategy': strategy,
            'seed_size': seed_size,
            'query_size': query_size,
            'rounds': rounds,
        })

        # precompute random baseline
        st.session_state.state['random_curve'] = run_random_baseline(st.session_state.state['X_pool'], st.session_state.state['y_pool'], st.session_state.state['X_test'], st.session_state.state['y_test'], classifier_name, seed_size, query_size, rounds, n_repeats=10)

        st.success('Initialized. Click "Next" to label one batch, or "Run all" to complete simulation.')

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    if col1.button('Next'):
        if 'X_pool' not in st.session_state.state:
            st.error('Click Initialize first')
        else:
            do_al_step(st.session_state.state)

    if col2.button('Run 10 steps'):
        if 'X_pool' not in st.session_state.state:
            st.error('Click Initialize first')
        else:
            before = len(st.session_state.state['labeled'])
            for _ in range(10):
                do_al_step(st.session_state.state, show_new=False)
            after = len(st.session_state.state['labeled'])
            added = max(0, after - before)
            st.success(f'Run 10 steps — labeled {added} new points (showing 1 representative)')
            if added > 0:
                last_idx = st.session_state.state['labeled'][-1]
                _display_labeled_samples(st.session_state.state, [last_idx])

    if col3.button('Run all'):
        if 'X_pool' not in st.session_state.state:
            st.error('Click Initialize first')
        else:
            before = len(st.session_state.state['labeled'])
            remaining = st.session_state.state['rounds'] - len(st.session_state.state['history'])
            for _ in range(remaining):
                do_al_step(st.session_state.state, show_new=False)
            after = len(st.session_state.state['labeled'])
            added = max(0, after - before)
            st.success(f'Run all — labeled {added} new points (showing 1 representative)')
            if added > 0:
                last_idx = st.session_state.state['labeled'][-1]
                _display_labeled_samples(st.session_state.state, [last_idx])

    if col4.button('Reset'):
        st.session_state.clear()
        try:
            st.experimental_rerun()
        except AttributeError:
            st.success('State cleared — please refresh the page to continue.')

    # Show status and plots
    if 'history' in st.session_state.state and len(st.session_state.state['history']) > 0:
        hist = pd.DataFrame(st.session_state.state['history'])
        # plot random baseline comparison only
        rnd = st.session_state.state.get('random_curve')
        if rnd is not None:
            df_rnd = pd.DataFrame({'round': list(range(len(rnd))), 'random_f1': rnd})
            combined = hist.merge(df_rnd, on='round', how='left')
            st.subheader('Comparison: Selected strategy vs Random baseline')
            st.line_chart(combined.set_index('round')[['f1_macro', 'random_f1']])

        st.subheader('Last step info')
        st.table(hist.tail(1))
    else:
        st.info('No AL steps run yet. Initialize and click Next.')


def _display_labeled_samples(state: dict, indices: list[int]):
    """Display images/labels for provided dataset indices (uses metadata if present)."""
    X_pool = state.get('X_pool')
    y_pool = state.get('y_pool')
    metadata_path = Path('data/synth_surface_defects/metadata.csv')
    if metadata_path.exists():
        meta = pd.read_csv(metadata_path)
        meta_map = {int(row['index']): row for _, row in meta.iterrows()}
        data_root = Path('data/synth_surface_defects')
        for idx in indices:
            row = meta_map.get(int(idx))
            if row is not None:
                img_path = data_root / row['filepath']
                label = row['label']
                if img_path.exists():
                    st.image(str(img_path), caption=f"Label: {label} (Index: {idx})", width=256)
                else:
                    st.write(f"Index: {idx}, Label: {label} (Image not found)")
            else:
                st.write(f"Index: {idx}, Label: {y_pool[int(idx)]} (Metadata not found)")
    else:
        for idx in indices:
            st.write(f"Index: {idx}, Label: {y_pool[int(idx)]}")


def do_al_step(state: dict, show_new: bool = True):
    X_pool = state['X_pool']
    y_pool = state['y_pool']
    X_test = state['X_test']
    y_test = state['y_test']
    labeled = state['labeled']
    unlabeled = state['unlabeled']
    clf = get_classifier(state['classifier_name'])

    # Train on labeled
    clf.fit(X_pool[labeled], y_pool[labeled])

    # Evaluate
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds, average='macro')
    acc = accuracy_score(y_test, preds)

    # Select next points
    qsize = state['query_size']
    if len(unlabeled) == 0:
        st.warning('No more unlabeled points')
        return

    # compute scores on unlabeled pool
    X_unl = X_pool[unlabeled]
    try:
        scores = strategy_scores(clf, X_unl, state['strategy'])
    except Exception:
        # fallback to random
        scores = np.random.random(len(unlabeled))

    # parse base strategy and whether to use diversity ("div")
    strat = state['strategy']
    use_div = strat.endswith('_div')
    base = strat.rsplit('_', 1)[0] if '_' in strat else strat

    # For margin strategy we returned top2 difference (top2 - top1), which is negative for small margin
    if base == 'margin':
        uncertainty_order = np.argsort(scores)  # small margin more uncertain
    else:
        uncertainty_order = np.argsort(-scores)

    # If diversity requested, perform greedy max-min selection from top candidates
    if use_div and len(uncertainty_order) > 0:
        pool_size = min(len(uncertainty_order), max(qsize * 10, qsize))
        candidate_local = uncertainty_order[:pool_size]
        selected_local = []
        # pick most uncertain first
        selected_local.append(candidate_local[0])
        while len(selected_local) < min(qsize, len(candidate_local)):
            rem = [c for c in candidate_local if c not in selected_local]
            # compute min distance to already selected for each remaining candidate
            best_idx = None
            best_min_dist = -1.0
            for c in rem:
                pt = X_unl[c]
                min_dist = min(np.linalg.norm(pt - X_unl[s]) for s in selected_local)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = c
            if best_idx is None:
                break
            selected_local.append(best_idx)
        chosen_local = selected_local
    else:
        chosen_local = uncertainty_order[:min(qsize, len(uncertainty_order))]

    chosen = [unlabeled[i] for i in chosen_local]
    for c in chosen:
        labeled.append(c)
        unlabeled.remove(c)


    if show_new:
        st.info('Newly labeled sample (representative):')
        if len(chosen) > 0:
            _display_labeled_samples(state, [chosen[-1]])
        else:
            st.write('No new samples labeled in this step.')

    # retrain after labeling
    clf.fit(X_pool[labeled], y_pool[labeled])
    preds2 = clf.predict(X_test)
    f1_after = f1_score(y_test, preds2, average='macro')
    acc_after = accuracy_score(y_test, preds2)

    round_no = len(state['history'])
    state['history'].append({'round': round_no, 'f1_macro': f1_after, 'accuracy': acc_after, 'n_labeled': len(labeled)})
    if show_new:
        st.success(f'Labeled {len(chosen)} new points — round {round_no} done. F1: {f1_after:.4f}')


if __name__ == '__main__':
    main()
