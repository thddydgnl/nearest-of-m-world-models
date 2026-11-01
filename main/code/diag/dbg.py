import torch, sys
from imlewm.model import Gen
from imlewm.wm import TokenData
torch.manual_seed(0)
data = TokenData("bounce_k4", "train", H=1); st = data.stats
gen = torch.Generator().manual_seed(0)
batches = [data.sample(128, gen) for _ in range(20)]
def run(method, zmode, steps=600, lr=2e-4):
    torch.manual_seed(0)
    m = Gen(method=method, n_heads=1).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.9))
    for s in range(steps):
        x, a = batches[s % 20]; x = st.norm(x); ctx, tgt = x[:, :3], x[:, 3]
        kw = {}
        if method == "imle":
            kw["z"] = torch.zeros(128, 256, device="cuda") + 1e-3 if zmode == "zero" else torch.randn(128, 256, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = ctx[:, -1] + m(ctx, a[:, 0], **kw)[-1].float()
        loss = (p - tgt).pow(2).mean()
        opt.zero_grad(); loss.backward()
        gn = {n: p_.grad.norm().item() for n, p_ in m.named_parameters() if n in ("stem.weight", "blocks.0.mod_s.weight", "heads.0.weight", "map.0.weight")}
        opt.step()
        if s % 150 == 0 or s == steps - 1:
            print(method, zmode, s, round(loss.item(), 4), {k: round(v, 4) for k, v in gn.items()}, flush=True)
run("reg", None)
run("imle", "rand")
run("imle", "zero")
