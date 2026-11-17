"""Turn per-frame posteriors P(D,H,A) into labels. Viterbi with switch penalties (per half)."""
import numpy as np

STATES = np.array(["D", "H", "A"])


def viterbi(logp: np.ndarray, pen_team: float, pen_dead: float, team_pen_t=None) -> np.ndarray:
    """logp (N,3) log-posteriors for D,H,A. team_pen_t: optional per-frame H<->A penalty."""
    n = len(logp)
    T = np.array([[0.0, -pen_dead, -pen_dead],
                  [-pen_dead, 0.0, -pen_team],
                  [-pen_dead, -pen_team, 0.0]])
    score = logp[0].copy()
    back = np.zeros((n, 3), np.int8)
    for t in range(1, n):
        if team_pen_t is not None:
            T[1, 2] = T[2, 1] = -team_pen_t[t]
        cand = score[:, None] + T
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + logp[t]
    path = np.empty(n, np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(proba: np.ndarray, period: np.ndarray, pen_team=8.0, pen_dead=8.0,
           change=None, alpha=0.0) -> np.ndarray:
    """change: optional per-frame P(possession change); lowers the H<->A penalty where high:
    pen_t = max(0.5, pen_team - alpha * logit(change))."""
    logp = np.log(np.clip(proba[:, :3], 1e-6, 1))
    tp = None
    if change is not None and alpha > 0:
        c = np.clip(change, 1e-4, 1 - 1e-4)
        tp = np.maximum(0.5, pen_team - alpha * np.log(c / (1 - c)))
    out = np.empty(len(proba), dtype="<U1")
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out[idx] = STATES[viterbi(logp[idx], pen_team, pen_dead, None if tp is None else tp[idx])]
    return out


def viterbi_md(logp: np.ndarray, pen_team: float, pen_enter_d: float, pen_exit_d: float,
               min_d: int = 1, min_t: int = 1) -> np.ndarray:
    """Viterbi with separate costs for entering / leaving a dead ball and minimum run lengths
    (min_d frames for D, min_t for H/A), via state expansion: each label is a chain of states
    that must be walked through before the label can change. Returns label indices (0 D, 1 H, 2 A)."""
    n = len(logp)
    m = [max(1, min_d), max(1, min_t), max(1, min_t)]
    lab = np.concatenate([np.full(k, i) for i, k in enumerate(m)])
    first = np.cumsum([0] + m[:-1])
    last = first + np.array(m) - 1
    S = len(lab)
    pen = np.array([[0.0, pen_exit_d, pen_exit_d],
                    [pen_enter_d, 0.0, pen_team],
                    [pen_enter_d, pen_team, 0.0]])
    NEG = -1e18
    score = np.full(S, NEG)
    score[first] = logp[0]
    back = np.zeros((n, S), np.int32)
    idx = np.arange(S)
    for t in range(1, n):
        new = np.full(S, NEG)
        arg = np.zeros(S, np.int32)
        # advance inside a chain (s -> s+1), not across chains
        inner = np.ones(S, bool)
        inner[first] = False
        new[inner] = score[idx[inner] - 1]
        arg[inner] = idx[inner] - 1
        # stay on the last state of a chain
        for i in range(3):
            if score[last[i]] > new[last[i]]:
                new[last[i]], arg[last[i]] = score[last[i]], last[i]
        # change label: from the last state of chain i to the first state of chain j
        for j in range(3):
            c = [score[last[i]] - pen[i, j] if i != j else NEG for i in range(3)]
            i = int(np.argmax(c))
            if c[i] > new[first[j]]:
                new[first[j]], arg[first[j]] = c[i], last[i]
        score = new + logp[t][lab]
        back[t] = arg
    path = np.empty(n, np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return lab[path]


def decode_md(proba, period, pen_team=8.0, pen_enter_d=24.0, pen_exit_d=24.0, min_d=1, min_t=1):
    logp = np.log(np.clip(proba[:, :3], 1e-6, 1))
    out = np.empty(len(proba), dtype="<U1")
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out[idx] = STATES[viterbi_md(logp[idx], pen_team, pen_enter_d, pen_exit_d, min_d, min_t)]
    return out
