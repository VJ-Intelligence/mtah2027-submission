"""Robustness: TCN trained on DFL only vs DFL + 10 SkillCorner, evaluated on the OTHER 10
SkillCorner matches (and DFL-only reference from the final models)."""
import numpy as np

from mtah import model, seqmodel, submit
from mtah.bench import period_runs
from mtah.decode import decode
from mtah.external import idsse, skillcorner
from mtah.scoring import score

sc = skillcorner.MATCH_IDS
train_sc, test_sc = sc[:10], sc[10:]
dfl = [model.sample(mid, 100 * i + s) for i, mid in enumerate(idsse.MATCH_IDS) for s in (0, 1, 2, 3)]
mix = dfl + [model.sample_sc(sid, 5000 + sc.index(sid)) for sid in train_sc]


def evaluate(m):
    seqmodel.DEVICE = "mps"
    f = []
    for sid in test_sc:
        X, y, names, per = model.sample_sc(sid, 5000 + sc.index(sid))
        p = seqmodel.predict_proba(m, X, per)[:, :3]
        f.append(score(np.array(list("DHA"))[y], decode(p, per, 8, 24), period_runs(per))["spell_f1"])
    return np.mean(f)


for name, parts in (("DFL only", dfl), ("DFL + 10 SC", mix)):
    m = seqmodel.train(parts, epochs=6, seed=7, log=lambda *_: None)
    print(f"{name:12s} -> held-out SkillCorner spellF1 {evaluate(m):.4f}", flush=True)
