"""Turn a make_body_snrext.py body into the paper's tab:eff and tab:cmp LaTeX tables. Usage: body_to_tables.py <body.txt> <out.tex>"""
import re, sys
body = open(sys.argv[1]).read().splitlines(); out = []
i = next(k for k, l in enumerate(body) if l.strip().startswith("SNR ")); hdr = body[i].split()[1:]
rows = [body[i + 1 + j].split() for j in range(4)]
out += [r"\begin{table}", r"\begin{center}", r"\begin{tabular}{|l|" + "c" * len(hdr) + "|}", r"\hline",
        r"$\rho_{\rm net}$ & " + " & ".join(hdr) + r"\\", r"\hline"]
for r in rows:
    run = r[0][:2] + r[0][2:].lower(); vals = [("--" if v == "-" else v) for v in r[1:1 + len(hdr)]]
    out.append(f"{run} & " + " & ".join(vals) + r"\\")
out += [r"\hline", r"\end{tabular}\caption{Recovery fraction at ${\rm FAR}<1\,{\rm yr^{-1}}$ versus injected H1+L1",
        r"network SNR. A dash marks a level absent for that run. Values are run averages", r"over the injected mass mix, not per-cell efficiencies.}\label{tab:eff}",
        r"\end{center}", r"\end{table}", ""]
txt = "\n".join(body)
tot = re.search(r"events with support\s+(\d+)\s+predicted\s+([\d.]+) \+/- ([\d.]+).*?\n\s+actual\s+(\d+)\s+difference\s+([+-][\d.]+) sigma", txt, re.S)
per = re.findall(r"(O3A|O3B|O4A|O4B): predicted\s+([\d.]+) \+/-\s+([\d.]+)\s+actual\s+(\d+)\s+([+-][\d.]+) sigma(?:\s+\(comparable (\d+)\))?", txt)
cmp_n = re.findall(r"(O3A|O3B|O4A|O4B).*?n=(\d+)", txt)  # not the comparable count; comparable per run is not in the body
out += [r"\begin{table}", r"\begin{center}", r"\begin{tabular}{|l|cccc|}", r"\hline", r"Run & Comparable & Found & Predicted & Difference\\", r"\hline"]
for run, pred, sd, act, sig, ncmp in per:
    out.append(f"{run[:2] + run[2:].lower()} & {ncmp or '[[n]]'} & {act} & ${pred}\\pm{sd}$ & ${sig}\\sigma$\\\\")
out += [r"\hline", f"Total & {tot.group(1)} & {tot.group(4)} & ${tot.group(2)}\\pm{tot.group(3)}$ & ${tot.group(5)}\\sigma$\\\\", r"\hline",
        r"\end{tabular}\caption{Simulation against data at each detection's own (source-frame mass, SNR) cell (see text).}\label{tab:cmp}", r"\end{center}", r"\end{table}"]
open(sys.argv[2], "w").write("\n".join(out) + "\n"); print("wrote", sys.argv[2]); print("\n".join(out))
