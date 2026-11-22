"""Revision r1, analysis e (DESIGN_r1.md section 4e): pit risk on the action sequences the planner selects.

replay: the CEM-MPC planner of plan.run (edge task, position cost, lam 0, 100 tasks, task seed 0) for one model,
        same code and random-number calls, recording at every planning step the real state, the selected
        T-step plan (the CEM mean, each action clipped to |a| <= 1 as executed) and the executed first action.
        Then, with fresh seeds:
        executed: for each episode, the executed action sequence replayed open-loop from the start state:
                  model pit probability (16 imagined rollouts, probe readout; criterion "last" = final imagined
                  frame in the pit, the planner's cost criterion, and "any" = any imagined frame, the risk
                  diagnostic's criterion) against the simulator's (256 rollouts; the pit is absorbing).
        selected: for every planning step, the selected T-step plan from the real state, same comparison.
riskdiag: the straight-push risk diagnostic of riskdiag.main (200 starts, 64 samples, 256 true rollouts, T = 5)
        with both criteria computed on the same samples.

usage: python -m imlewm.r1_planrisk replay --model e13_push_imle16_s0 --out /workspace/nom/out/e_planrisk
       python -m imlewm.r1_planrisk riskdiag --models e13_push_imle16_s0 ... [--nfe 2] --out ...
"""
import argparse
import os
import time

import numpy as np
import torch

from . import r1_util
from .common import HI, ICE, LO, PIT, S_PUSH, DinoEncoder, in_rect, push_step, render
from .plan import tasks
from .probe import load_probe
from .riskdiag import in_pit_t
from .wm import Stats, load_model, predict


def push_step_vec(p, a, rng):
    """Vectorized common.push_step: p (n,2), a (n,2) -> next positions (pit absorbing, ice 3 modes, else 2)."""
    p = np.asarray(p, float); a = np.asarray(a, float)
    in_pit = (p[:, 0] >= PIT[0]) & (p[:, 0] < PIT[2]) & (p[:, 1] >= PIT[1]) & (p[:, 1] < PIT[3])
    on_ice = (p[:, 0] >= ICE[0]) & (p[:, 0] < ICE[2]) & (p[:, 1] >= ICE[1]) & (p[:, 1] < ICE[3])
    k3 = rng.integers(3, size=len(p)); k2 = rng.integers(2, size=len(p))
    ang = np.where(on_ice, np.deg2rad(np.array([-60.0, 0.0, 60.0]))[k3], np.deg2rad(np.array([-20.0, 20.0]))[k2])
    c, s = np.cos(ang), np.sin(ang)
    v = np.stack([c * a[:, 0] - s * a[:, 1], s * a[:, 0] + c * a[:, 1]], 1)
    q = np.clip(p + S_PUSH * v, LO, HI)
    return np.where(in_pit[:, None], p, q)


def sim_pit(p0, acts, n_true, rng):
    """p0 (2,), acts (L,2) -> share of n_true simulator rollouts that end in the pit."""
    p = np.repeat(np.asarray(p0, float)[None], n_true, 0)
    for a in acts:
        p = push_step_vec(p, np.repeat(np.asarray(a, float)[None], n_true, 0), rng)
    return float(((p[:, 0] >= PIT[0]) & (p[:, 0] < PIT[2]) & (p[:, 1] >= PIT[1]) & (p[:, 1] < PIT[3])).mean())


@torch.no_grad()
def plan_replay(model_name, lam=0.0, n_eps=100, seed=0, T=5, N=64, iters=3, n_elite=8, m_p=8, budget=16,
                spread="std", chunk=512, cost="pos", task="edge", channels_last=False):
    """plan.run with a record of every planning step (the planner itself is unchanged; channels_last only changes
    the memory layout of the model's weights, about 1.3x faster on the A40, same computation up to bf16 rounding)."""
    t_start = time.time()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    st = Stats("push")
    enc = DinoEncoder()
    model = load_model(f"{r1_util.RUNS}/{model_name}/ckpt.pt")[0]
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if model.method == "reg":
        m_p = 1
    probe = load_probe("push")
    starts, goals = tasks(n_eps, rng, task)
    goal_tok = st.norm(torch.from_numpy(enc(render(goals, "push"))).cuda())
    goal_pos_t = torch.from_numpy(goals).float().cuda()
    pos = starts.copy()
    ctx = st.norm(torch.from_numpy(enc(render(pos, "push"))).cuda())[:, None].repeat(1, 3, 1, 1)
    status = np.zeros(n_eps, int)
    steps_used = np.zeros(n_eps, int)
    env_rng = np.random.default_rng(seed + 1)
    rec, hist = [], [[starts[e].copy()] for e in range(n_eps)]
    for t in range(budget):
        act = np.where(status == 0)[0]
        if len(act) == 0:
            break
        E = len(act)
        mu = torch.zeros(E, T, 2, device="cuda"); sd = torch.full((E, T, 2), 0.6, device="cuda")
        for it in range(iters):
            cand = (mu[:, None] + sd[:, None] * torch.randn(E, N, T, 2, device="cuda"))
            nrm = cand.norm(dim=-1, keepdim=True).clamp(min=1.0)
            cand = cand / nrm
            aa = cand[:, :, None].expand(E, N, m_p, T, 2).reshape(-1, T, 2)
            eid = torch.from_numpy(act).cuda().repeat_interleave(N * m_p)
            ds = []
            for i in range(0, len(aa), chunk):
                e = eid[i:i + chunk]
                c = ctx[e]
                for h in range(T):
                    s = predict(model, c, aa[i:i + chunk, h], 1, nfe=2, sigma_r=st.sigma_r)[:, 0]
                    c = torch.cat([c[:, 1:], s[:, None]], 1)
                if cost == "pos":
                    ds.append((probe(c[:, -1]) - goal_pos_t[e]).norm(dim=-1))
                else:
                    ds.append((c[:, -1] - goal_tok[e]).pow(2).mean((-1, -2)).sqrt())
            d = torch.cat(ds).view(E, N, m_p)
            if m_p > 1:
                spr = d.std(2) if spread == "std" else d.var(2)
            else:
                spr = torch.zeros_like(d[:, :, 0])
            J = d.mean(2) + lam * spr
            el = J.topk(n_elite, dim=1, largest=False).indices
            elite = cand[torch.arange(E)[:, None], el]
            mu, sd = elite.mean(1), elite.std(1) + 0.05
        plan_sel = (mu / mu.norm(dim=-1, keepdim=True).clamp(min=1.0)).cpu().numpy()  # (E,T,2) as executed
        a0 = mu[:, 0]
        a0 = (a0 / a0.norm(dim=-1, keepdim=True).clamp(min=1.0)).cpu().numpy()
        newpos = np.stack([push_step(pos[e], a0[i], env_rng)[0] for i, e in enumerate(act)])
        for i, e in enumerate(act):
            h_ = hist[e]
            rec.append({"ep": int(e), "t": int(t), "ctx_pos": [h_[max(0, len(h_) - 3 + j)].tolist() for j in range(3)],
                        "pos": pos[e].tolist(), "plan": plan_sel[i].tolist(), "a0": a0[i].tolist(),
                        "next_pos": newpos[i].tolist()})
        pos[act] = newpos
        steps_used[act] += 1
        tok = st.norm(torch.from_numpy(enc(render(newpos, "push"))).cuda())
        ctx[act] = torch.cat([ctx[act][:, 1:], tok[:, None]], 1)
        for i, e in enumerate(act):
            hist[e].append(pos[e].copy())
            if in_rect(pos[e], PIT):
                status[e] = -1
            elif np.linalg.norm(pos[e] - goals[e]) < 5.0:
                status[e] = 1
        print(f"{model_name} step {t} active {E} elapsed {time.time() - t_start:.0f}s", flush=True)
    summ = {"lam": lam, "model": model_name, "n_eps": n_eps, "m_p": m_p, "wall_s": round(time.time() - t_start, 1),
            "success": float((status == 1).mean()), "pit": float((status == -1).mean()),
            "timeout": float((status == 0).mean()),
            "steps_success": float(steps_used[status == 1].mean()) if (status == 1).any() else None}
    return summ, rec, starts, goals, status


@torch.no_grad()
def imagined_pit(model, st, probe, enc, ctx_pos, acts, n, nfe, bs=64):
    """ctx_pos (R,3,2) real context positions, acts list of (L_r,2) -> per row (pit_last, pit_any) over n samples."""
    out_last, out_any = np.zeros(len(ctx_pos)), np.zeros(len(ctx_pos))
    Ls = np.array([len(a) for a in acts])
    for L in np.unique(Ls):
        rows = np.where(Ls == L)[0]
        for r0 in range(0, len(rows), bs):
            rr = rows[r0:r0 + bs]
            cp = np.asarray(ctx_pos)[rr].reshape(-1, 2)
            c = st.norm(torch.from_numpy(enc(render(cp, "push"))).cuda()).view(len(rr), 3, 256, 384)
            c = c.repeat_interleave(n, 0)
            A = torch.from_numpy(np.stack([acts[i] for i in rr])).float().cuda().repeat_interleave(n, 0)
            fell = torch.zeros(len(c), dtype=torch.bool, device="cuda")
            for h in range(L):
                s_ = predict(model, c, A[:, h], 1, nfe, st.sigma_r)[:, 0]
                inp = in_pit_t(probe(s_))
                fell |= inp
                c = torch.cat([c[:, 1:], s_[:, None]], 1)
            out_last[rr] = inp.view(len(rr), n).float().mean(1).cpu().numpy()
            out_any[rr] = fell.view(len(rr), n).float().mean(1).cpu().numpy()
    return out_last, out_any


def compare(model_last, model_any, true):
    def one(mp):
        d = mp - true
        return {"model_mean": float(mp.mean()), "true_mean": float(true.mean()), "mean_diff": float(d.mean()),
                "abs_err": float(np.abs(d).mean()),
                "corr": float(np.corrcoef(mp, true)[0, 1]) if mp.std() > 0 and true.std() > 0 else None,
                "share_under": float((d < -0.05).mean()), "share_over": float((d > 0.05).mean())}
    return {"n": len(true), "last": one(model_last), "any": one(model_any)}


def replay_cmd(model_name, out, n_eps=100, n_samp=16, n_true=256, channels_last=False):
    path = f"{out}/{model_name}_replay.json"
    if os.path.exists(path):
        print("skip", path); return
    summ, rec, starts, goals, status = plan_replay(model_name, n_eps=n_eps, channels_last=channels_last)
    summ["channels_last"] = channels_last
    print("replay", summ, flush=True)
    r1_util.write_json({"summary": summ, "records": rec, "status": status.tolist(), "starts": starts.tolist(),
                        "goals": goals.tolist()}, f"{out}/{model_name}_records.json")
    st = Stats("push"); probe = load_probe("push"); enc = DinoEncoder()
    model = load_model(f"{r1_util.RUNS}/{model_name}/ckpt.pt")[0]
    nfe = 2  # the planner's sampling steps (FM-2); ignored by IMLE-WM and regression
    torch.manual_seed(1000); srng = np.random.default_rng(1000)
    # selected T-step plans from the real state of every planning step
    plans = [np.asarray(r["plan"]) for r in rec]
    ml, ma = imagined_pit(model, st, probe, enc, [r["ctx_pos"] for r in rec], plans, n_samp, nfe)
    tp = np.array([sim_pit(r["pos"], p, n_true, srng) for r, p in zip(rec, plans)])
    sel = compare(ml, ma, tp)
    on_ice = np.array([in_rect(np.asarray(r["pos"]), ICE) for r in rec])
    risky = tp > 0
    sel["on_ice"] = compare(ml[on_ice], ma[on_ice], tp[on_ice]) if on_ice.any() else None
    sel["true_pit_positive"] = compare(ml[risky], ma[risky], tp[risky]) if risky.any() else None
    # executed sequences, open loop from the start state
    eps = sorted(set(r["ep"] for r in rec))
    seqs = [np.array([r["a0"] for r in rec if r["ep"] == e]) for e in eps]
    sctx = [[starts[e].tolist()] * 3 for e in eps]
    torch.manual_seed(1001)
    el, ea = imagined_pit(model, st, probe, enc, sctx, seqs, n_samp, nfe)
    et = np.array([sim_pit(starts[e], s, n_true, srng) for e, s in zip(eps, seqs)])
    exe = compare(el, ea, et)
    st_ep = status[np.array(eps)]
    for name, msk in (("pit_episodes", st_ep == -1), ("success_episodes", st_ep == 1), ("timeout_episodes", st_ep == 0)):
        exe[name] = compare(el[msk], ea[msk], et[msk]) if msk.sum() > 1 else None
    res = {"summary": summ, "n_samp": n_samp, "n_true": n_true, "nfe": nfe if model.method == "fm" else 1,
           "selected": sel, "executed": exe,
           "per_record": {"model_last": ml.round(4).tolist(), "model_any": ma.round(4).tolist(), "true": tp.round(4).tolist()},
           "per_episode": {"ep": eps, "len": [len(s) for s in seqs], "status": st_ep.tolist(),
                           "model_last": el.round(4).tolist(), "model_any": ea.round(4).tolist(), "true": et.round(4).tolist()}}
    r1_util.write_json(res, path)
    print("selected", {k: sel[k] for k in ("last", "any")}, "\nexecuted", {k: exe[k] for k in ("last", "any")}, flush=True)


@torch.no_grad()
def riskdiag_both(models, nfe=2, n=200, n_samp=64, n_true=256, T=5, seed=0):
    """riskdiag.main with the fall counted at the last imagined frame (planner cost) and at any frame."""
    rng = np.random.default_rng(seed)
    starts, goals = tasks(n, rng, "edge")
    d = goals - starts
    act = d / np.linalg.norm(d, axis=1, keepdims=True)
    sim_rng = np.random.default_rng(seed + 7)
    true_pit = np.zeros(n)
    for i in range(n):
        fallen = 0
        for _ in range(n_true):
            p = starts[i].copy()
            for _h in range(T):
                p = push_step(p, act[i], sim_rng)[0]
            fallen += in_rect(p, PIT)
        true_pit[i] = fallen / n_true
    ice = np.stack([rng.uniform(22, 34, n), rng.uniform(17, 21, n)], 1)
    ice_true = np.mean([[in_rect(push_step(q, np.array([1.0, 0.0]), sim_rng)[0], PIT) for _ in range(64)] for q in ice])
    st = Stats("push"); enc = DinoEncoder(); probe = load_probe("push")
    ctx0 = st.norm(torch.from_numpy(enc(render(starts, "push"))).cuda())[:, None].repeat(1, 3, 1, 1)
    ctx_ice = st.norm(torch.from_numpy(enc(render(ice, "push"))).cuda())[:, None].repeat(1, 3, 1, 1)
    a_t = torch.from_numpy(act).float().cuda()
    out = {"n": n, "T": T, "nfe": nfe, "true_pit_mean": float(true_pit.mean()), "ice_true_pit_share": float(ice_true),
           "true_pit": true_pit.tolist(), "models": {}}
    for name in models:
        torch.manual_seed(seed)  # same random-number calls as riskdiag.main, so "any" reproduces its model_pit
        model, _ = load_model(f"{r1_util.RUNS}/{name}/ckpt.pt")
        pl, pa, ice_hits = [], [], []
        for i0 in range(0, n, 16):
            c = ctx0[i0:i0 + 16].repeat_interleave(n_samp, 0)
            aa = a_t[i0:i0 + 16].repeat_interleave(n_samp, 0)
            fell = torch.zeros(len(c), dtype=torch.bool, device="cuda")
            for h in range(T):
                s_ = predict(model, c, aa, 1, nfe, st.sigma_r)[:, 0]
                inp = in_pit_t(probe(s_))
                fell |= inp
                c = torch.cat([c[:, 1:], s_[:, None]], 1)
            pl.append(inp.view(-1, n_samp).float().mean(1).cpu()); pa.append(fell.view(-1, n_samp).float().mean(1).cpu())
            ci = ctx_ice[i0:i0 + 16].repeat_interleave(n_samp, 0)
            right = torch.tensor([[1.0, 0.0]], device="cuda").expand(len(ci), 2)
            s1 = predict(model, ci, right, 1, nfe, st.sigma_r)[:, 0]
            ice_hits.append(in_pit_t(probe(s1)).float().cpu())
        pl, pa = torch.cat(pl).numpy(), torch.cat(pa).numpy()
        r = compare(pl, pa, true_pit)
        r["method"] = model.method
        r["ice_one_step_pit_share"] = torch.cat(ice_hits).mean().item()
        r["model_pit_last"] = pl.round(4).tolist(); r["model_pit_any"] = pa.round(4).tolist()
        out["models"][name] = r
        print(name, nfe, {k: r[k] for k in ("last", "any")}, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["replay", "riskdiag"])
    ap.add_argument("--model")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--nfe", type=int, default=2)
    ap.add_argument("--eps", type=int, default=100)
    ap.add_argument("--channels_last", action="store_true", help="model weights in channels_last layout (faster)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "replay":
        replay_cmd(a.model, a.out, a.eps, channels_last=a.channels_last)
    else:
        p = f"{a.out}/riskdiag_both_nfe{a.nfe}.json"
        if os.path.exists(p):
            print("skip", p)
        else:
            r1_util.write_json(riskdiag_both(a.models, a.nfe), p)
