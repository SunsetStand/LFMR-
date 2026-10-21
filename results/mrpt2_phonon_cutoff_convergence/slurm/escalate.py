"""Resubmit memory-capped MRPT2 points at the next lean memory tier.

Exits 0 when every tracked point is converged, 1 on any unexpected state.
"""
import json
import os
import subprocess
import sys
import time

R = "/DFS-L/DATA/cui/chenxw25/lfmr_runs/mrpt2_mscan"
CODE = f"{R}/code_w4"
TIERS = [
    None,  # whatever is already running
    dict(cpus=40, mem="120G", part="sib2.9,has2.5,brd2.4"),
    dict(cpus=40, mem="245G", part="brd2.4,ilg2.3"),
    dict(cpus=64, mem="490G", part="ilg2.3"),
]
LOG = f"{R}/escalate.log"


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def queued():
    out = subprocess.run(
        ["squeue", "-u", os.environ["USER"], "-h", "-o", "%i %j"], capture_output=True, text=True
    ).stdout
    return {line.split()[1]: line.split()[0] for line in out.splitlines() if line.strip()}


def sacct_state(job):
    out = subprocess.run(
        ["sacct", "-j", job, "-n", "-X", "-o", "State"], capture_output=True, text=True
    ).stdout
    return out.strip().split()[0] if out.strip() else "UNKNOWN"


def submit(name, case, omega, alpha, tier):
    t = TIERS[tier]
    cmd = [
        "sbatch", "-J", name, "-c", str(t["cpus"]), f"--mem={t['mem']}", "-p", t["part"],
        "--time=7-00:00:00", f"{R}/run_point.sbatch", CODE, case, omega, alpha,
        "--algorithm", "lean",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=R)
    if out.returncode:
        log(f"PROBLEM sbatch failed for {name}: {out.stderr.strip()}")
        sys.exit(1)
    job = out.stdout.split()[-1]
    log(f"submitted {name} tier={tier} {t} job={job}")
    return job


TIER_START = {"m_w0.5_a2.0": 1, "m_w0.5_a2.4": 1}  # already resubmitted at tier 1
points = {}  # name -> dict(case, omega, alpha, tier, job)
for spec in sys.argv[1:]:
    case, omega, alpha = spec.split(":")
    prefix = "s" if case == "single" else "m"
    name = f"{prefix}_w{omega}_a{alpha}"
    points[name] = dict(case=case, omega=omega, alpha=alpha, tier=0, job=None)
q = queued()
for name, p in list(points.items()):
    p["job"] = q.get(name)
    p["tier"] = TIER_START.get(name, 0)
    if p["job"] is None:
        path = f"{R}/{p['case']}/w{p['omega']}_a{p['alpha']}.json"
        if os.path.exists(path) and json.load(open(path)).get("stop_reason") == "converged":
            log(f"CONVERGED {name} (already finished)")
            del points[name]
        else:
            log(f"PROBLEM {name} is not in the queue at start")
            sys.exit(1)
log(f"tracking {sorted(points)}")
while points:
    time.sleep(120)
    q = queued()
    for name in list(points):
        p = points[name]
        if name in q:
            continue
        state = sacct_state(p["job"])
        path = f"{R}/{p['case']}/w{p['omega']}_a{p['alpha']}.json"
        status = json.load(open(path)) if os.path.exists(path) else {}
        reason = status.get("stop_reason") or ""
        fresh = os.path.exists(path) and os.path.getmtime(path) > time.time() - 3 * 86400
        if state == "COMPLETED" and reason == "converged" and fresh:
            log(f"CONVERGED {name} M*={status['converged_M']} job={p['job']}")
            del points[name]
        elif (
            (state == "COMPLETED" and reason.startswith("memory")) or state == "OUT_OF_MEMORY"
        ) and p["tier"] + 1 < len(TIERS):
            log(f"{name} stopped: state={state} {reason}")
            p["tier"] += 1
            p["job"] = submit(name, p["case"], p["omega"], p["alpha"], p["tier"])
        else:
            log(f"PROBLEM {name} job={p['job']} state={state} stop_reason={reason!r}")
            sys.exit(1)
log("all tracked points converged")
