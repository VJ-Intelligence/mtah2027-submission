"""Out-of-domain check: final models (trained on DFL only) on 20 SkillCorner matches
(A-League, broadcast tracking, different possession semantics). Compared with the baseline."""
import numpy as np

from mtah import baselines, model, seqmodel, submit
from mtah.bench import period_runs
from mtah.decode import decode
from mtah.external import skillcorner
from mtah.scoring import score

seqmodel.DEVICE = "cpu"
booster, tcns = submit.load_models()
ens, base = [], []
for j, sid in enumerate(skillcorner.MATCH_IDS):
    X, y, names, per = model.sample_sc(sid, 5000 + j)
    gt = np.array(list("DHA"))[y]
    ps = [model.predict_proba(booster, X, names)] + [seqmodel.predict_proba(t, X, per)[:, :3] for t in tcns]
    lab = decode(np.mean(ps, 0), per, submit.PEN_TEAM, submit.PEN_DEAD)
    ens.append(score(gt, lab, period_runs(per))["spell_f1"])
    m = dict(skillcorner.load(sid))
    m["home"] = np.where(m["home_detected"][..., None], m["home"], np.nan)
    m["away"] = np.where(m["away_detected"][..., None], m["away"], np.nan)
    m["ball"] = np.where(m["ball_detected"][:, None], m["ball"], np.nan)
    m["label"] = model.sc_labels(m["label"], m["period"])
    d = model.degrade(m, seed=5000 + j)
    base.append(score(gt, baselines.predict(d, 50), period_runs(per))["spell_f1"])
    print(f"{sid}: ensemble {ens[-1]:.4f}  baseline50 {base[-1]:.4f}", flush=True)
print(f"MEAN ensemble {np.mean(ens):.4f} (sd {np.std(ens):.4f}) | baseline50 {np.mean(base):.4f} on {len(ens)} SkillCorner matches")
