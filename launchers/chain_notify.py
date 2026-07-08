"""Shared notify + result-file helper for the MADGRAV chain. PORTABLE by design:
the result text is ALWAYS written to a file (and printed); email is attempted ONLY if NOTIFY_EMAIL
is set (site.conf). So a stranger who runs the pipeline sees exactly what we see in the email —
in <out>/SUMMARY.txt (final) / STATUS.txt (progress) and the job log — with zero configuration.

Two delivery modes (env SMTP_MODE):
  smtp-auth : SMTP_HOST:SMTP_PORT + STARTTLS + login(SMTP_USER,SMTP_PASS) — works anywhere with creds.
  direct-mx : raw port-25 to each host in SMTP_HOST (comma list), no auth (needs outbound :25 allowed).
Config (env, exported by run_chain.sh from site.conf):
  NOTIFY_EMAIL NOTIFY_FROM SMTP_MODE SMTP_HOST SMTP_PORT SMTP_USER SMTP_PASS SMTP_HELO
"""
import os, smtplib
from email.message import EmailMessage


def cfg():
    return dict(
        to=os.environ.get("NOTIFY_EMAIL", "").strip(),
        frm=os.environ.get("NOTIFY_FROM", "madgrav@localhost"),
        mode=os.environ.get("SMTP_MODE", "smtp-auth"),
        hosts=[h.strip() for h in os.environ.get("SMTP_HOST", "").split(",") if h.strip()],
        port=int(os.environ.get("SMTP_PORT", "587") or 587),
        user=os.environ.get("SMTP_USER", ""),
        pw=os.environ.get("SMTP_PASS", ""),
        helo=os.environ.get("SMTP_HELO", "localhost"),
    )


def write_file(out_dir, filename, text):
    try:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, filename)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[notify] wrote {p}", flush=True)
        return p
    except Exception as e:
        print(f"[notify] file write FAIL: {e}", flush=True)
        return None


def send_email(subject, body, html=None, c=None):
    c = c or cfg()
    if not c["to"]:
        print("[notify] NOTIFY_EMAIL not set -> file/log only (no email)", flush=True)
        return False
    m = EmailMessage(); m["Subject"] = subject; m["From"] = c["frm"]; m["To"] = c["to"]; m.set_content(body)
    if html:
        m.add_alternative(html, subtype="html")
    try:
        if c["mode"] == "smtp-auth":
            host = c["hosts"][0] if c["hosts"] else "localhost"
            s = smtplib.SMTP(host, c["port"], timeout=30); s.ehlo(c["helo"])
            try: s.starttls(); s.ehlo(c["helo"])
            except Exception: pass
            if c["user"]: s.login(c["user"], c["pw"])
            s.send_message(m); s.quit()
            print(f"[notify] email sent (smtp-auth via {host})", flush=True); return True
        else:  # direct-mx
            for mx in (c["hosts"] or ["localhost"]):
                try:
                    s = smtplib.SMTP(mx, c["port"], timeout=30); s.ehlo(c["helo"]); s.send_message(m); s.quit()
                    print(f"[notify] email sent (direct-mx via {mx})", flush=True); return True
                except Exception as e:
                    print(f"[notify] direct-mx FAIL via {mx}: {e}", flush=True)
    except Exception as e:
        print(f"[notify] email FAIL: {e}", flush=True)
    return False


def deliver(out_dir, filename, subject, body, html=None):
    """ALWAYS write <out_dir>/<filename> with `body`; email the same content if configured.
    Returns (file_path_or_None, emailed_bool)."""
    p = write_file(out_dir, filename, body)
    e = send_email(subject, body, html=html)
    return p, e
