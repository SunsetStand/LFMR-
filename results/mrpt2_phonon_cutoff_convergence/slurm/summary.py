import csv, glob, json, os, sys
R = os.path.dirname(os.path.abspath(__file__))
for case in ["single", "multi"]:
    for f in sorted(glob.glob(f"{R}/{case}/*.csv"), key=lambda p: tuple(float(x[1:]) for x in os.path.basename(p)[:-4].split('_'))):
        rows = list(csv.DictReader(open(f)))
        name = os.path.basename(f)[:-4]
        out = []
        for cut in ("1e-08", "1e-10"):
            ok = {}
            fail = []
            for r in rows:
                if abs(float(r['external_cut']) - float(cut)) > 1e-20: continue
                if r["status"] == "ok": ok[int(r["max_total"])] = float(r["corrected_energy"])
                else: fail.append(int(r["max_total"]))
            Ms = sorted(ok)
            deltas = " ".join(f"{M}:{abs(ok[M]-ok[M-1]):.1e}" for M in Ms if M-1 in ok)
            out.append(f"  cut {cut}: Mmax_ok={Ms[-1] if Ms else None} E={ok[Ms[-1]] if Ms else float('nan'):.10f} fail={fail} | {deltas}")
        js = f[:-4] + ".json"
        st = json.load(open(js)) if os.path.exists(js) else {}
        print(f"{case} {name} stop={st.get('stop_reason')} conv={st.get('converged_M')}")
        print("\n".join(out))
