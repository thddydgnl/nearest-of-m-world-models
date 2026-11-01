"""Is IMLE training data-bound? Time CPU gather+transfer vs GPU-resident gather vs one train step."""
import time, torch
from imlewm.wm import TokenData, K_CTX
from imlewm.model import Gen

data = TokenData("bounce_div_k4", "train", H=1)
st = data.stats
gen = torch.Generator().manual_seed(0)
B = 128

def t(f, n=20):
    f(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t0) / n * 1000

cpu_ms = t(lambda: data.sample(B, gen))
tok_gpu = data.tok.cuda()  # fp16, ~8GB
valid_gpu = data.valid.cuda()
def gpu_sample():
    i = valid_gpu[torch.randint(len(valid_gpu), (B,), device="cuda")]
    fi = (i[:, None] + torch.arange(-K_CTX + 1, 2, device="cuda")[None]).reshape(-1)
    return tok_gpu[fi].view(B, K_CTX + 1, 256, 384)
gpu_ms = t(gpu_sample)

x, a = data.sample(B, gen)
for C, nb, tag in [(256, 8, "7M"), (384, 12, "19M")]:
    for m in [4, 16]:
        model = Gen(method="imle", C=C, nblocks=nb, n_heads=1).cuda()
        opt = torch.optim.AdamW(model.parameters(), 1e-4)
        def step():
            xx = st.norm(x); ctx, tgt = xx[:, :K_CTX], xx[:, K_CTX]
            z = torch.randn(B, m, model.zdim, device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                c = ctx.repeat_interleave(m, 0)
                s = c[:, -1] + model(c, a[:, 0].repeat_interleave(m, 0), z=z.reshape(B * m, -1))[-1].float()
                j = (s - tgt.repeat_interleave(m, 0)).pow(2).mean((1, 2)).view(B, m).argmin(1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = ctx[:, -1] + model(ctx, a[:, 0], z=z[torch.arange(B), j])[-1].float()
                loss = (p - tgt).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"compute step {tag} m={m}: {t(step, 10):.0f} ms", flush=True)
print(f"CPU gather+transfer: {cpu_ms:.0f} ms/batch   GPU-resident gather: {gpu_ms:.1f} ms/batch")
