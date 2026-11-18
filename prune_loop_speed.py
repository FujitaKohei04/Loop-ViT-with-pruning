"""GPU throughput of rot/joint_mc (merged into results_mc.json) and CPU throughput of joint_mc vs fewer steps (cpu_speed.json).

    uv run python prune_loop_speed.py --checkpoint <best.pth> --run outputs/bvit_pruning_loop/<run>
"""
import argparse, copy, json, statistics, time
from pathlib import Path
import torch
from dl_dataset.registry import get_spec
from prune_compute_bvit import Meter
from pruning import StepLimited
from pruning.channels import compact_channels, rotate_heads
from models.bViT.schedule import LoopSchedule
from train_bvit import load_model
from wandb_results import add_wandb_args, upload_results

p = argparse.ArgumentParser()
p.add_argument("--checkpoint"); p.add_argument("--run")
p.add_argument("--methods", nargs="+", default=["joint_mc"])
p.add_argument("--max-steps", type=int, default=None, help="the run pruned a model that stops after this many steps")
p.add_argument("--threads", type=int, default=4); p.add_argument("--cpu-batch", type=int, default=32)
add_wandb_args(p)
a = p.parse_args()
run = Path(a.run)
gpu = torch.device("cuda")
model, ck = load_model(a.checkpoint, use_ema=True, device=gpu); model.eval()
img = get_spec(ck["dataset"], "./data").img_size
rot = copy.deepcopy(model)
if a.max_steps:
    rot.schedule = LoopSchedule.parse(f"0x{a.max_steps}")
rotate_heads(rot)
saved = torch.load(run / "masks_mc.pt")
res = json.loads((run / "results_mc.json").read_text())
margs = argparse.Namespace(rounds=5, infer_batch=256, iters=15, latency_iters=30)
meter = Meter(model, [], img, gpu, margs)
for r in res["records"]:
    if r["setting"] == "rot" and r["method"] in a.methods and "rel_throughput" not in r:
        e = saved[("rot", r["method"], r["target"])]
        st = meter.compute(compact_channels(rot, e["masks"], means=e["means"]).eval())
        r.update({k: v for k, v in st.items() if k.startswith(("rel_", "img_per"))})
        print("gpu", r["method"], r["target"], round(st["rel_throughput"], 2), flush=True)
(run / "results_mc.json").write_text(json.dumps(res, indent=1))

torch.set_num_threads(a.threads)
cpu = torch.device("cpu")
model, rot = model.to(cpu), rot.to(cpu)
cands = {("unpruned", None): model}
for t in sorted({k[2] for k in saved if k[:2] == ("rot", "joint_mc")}, reverse=True):
    e = saved[("rot", "joint_mc", t)]
    cands[("joint_mc", t)] = compact_channels(rot, e["masks"], means=e["means"]).eval()
for s in range(model.schedule.num_steps - 1, 2, -1):
    cands[("steps", s)] = StepLimited(model, s).eval()
x = torch.randn(a.cpu_batch, 3, img, img)
def speed(m, n=4):
    with torch.inference_mode():
        m(x); t0 = time.perf_counter()
        for _ in range(n): m(x)
    return n * a.cpu_batch / (time.perf_counter() - t0)
ratios = {k: [] for k in cands}
for _ in range(5):
    base = speed(model)
    for k, m in cands.items():
        ratios[k].append(speed(m) / base)
out = [{"method": k[0], "target" if k[0] != "steps" else "steps": k[1], "rel_throughput_cpu": statistics.median(v)}
       for k, v in ratios.items()]
(run / "cpu_speed.json").write_text(json.dumps({"threads": a.threads, "batch": a.cpu_batch, "records": out}, indent=1))
for o in out: print("cpu", o)
upload_results(a, job_type="prune_loop_speed", group=run.name, files=[run / "results_mc.json", run / "cpu_speed.json"])
