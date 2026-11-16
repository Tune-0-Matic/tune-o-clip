"""Credit guard: tune-o-clip never pays per token. All judging / directing runs on local Ollama (this PC, the mini PC,
or the RunPod Pod's own Ollama). This scan fails the test suite if runtime code ever gains a paid LLM SDK, its API key,
or a web host that isn't on the list below (a new LLM endpoint of any provider shows up as an unknown host).
Run: python backend/test_no_paid_llm_calls.py
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = re.compile(r"\.(py|pyw|gd|sh|bat|html|js)$|requirements[^/]*\.txt$")
PAID = re.compile(r"\b(anthropic|openai|cohere|mistralai|google\.generativeai|google\.genai|groq|"  # Together / Replicate: by API domain below (plain English words)
                  r"ANTHROPIC_API_KEY|OPENAI_API_KEY|CLAUDE_API_KEY|GEMINI_API_KEY|GROQ_API_KEY)\b"
                  r"|api\.anthropic\.com|api\.openai\.com|generativelanguage\.googleapis\.com|api\.groq\.com|api\.mistral\.ai"
                  r"|api\.cohere\.|api\.together\.|api\.replicate\.com|openrouter\.ai", re.I)
# every web host the runtime may talk to: local Ollama, the platforms it clips from / posts to, RunPod, installers
HOSTS = {"127.0.0.1", "localhost", "www.youtube.com", "youtube.com", "www.googleapis.com", "oauth2.googleapis.com",
         "accounts.google.com", "api.twitch.tv", "id.twitch.tv", "gql.twitch.tv", "www.twitch.tv", "dev.twitch.tv", "clips.twitch.tv",
         "open.tiktokapis.com", "www.tiktok.com", "rest.runpod.io", "api.runpod.io", "ollama.com", "github.com", "www.w3.org",
         "discord.com", "discordapp.com", "youtu.be", "open-upload.tiktokapis.com"}
URL = re.compile(r"https?://([A-Za-z0-9.-]+)")


def runtime_files():
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True)
    me = Path(__file__).resolve()
    return [ROOT / f for f in out.stdout.splitlines()
            if RUNTIME.search(f) and (ROOT / f).is_file() and (ROOT / f).resolve() != me]


def scan(files):
    """-> ([(file, line no, match)] paid-LLM hits, [(file, host)] unknown hosts)"""
    paid, hosts = [], []
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for m in PAID.finditer(line):
                paid.append((f.relative_to(ROOT).as_posix(), n, m.group(0)))
            for h in [] if f.name.startswith("test_") else URL.findall(line):  # tests use made-up fixture hosts
                if h.lower().rstrip(".") not in HOSTS:
                    hosts.append((f.relative_to(ROOT).as_posix(), h))
    return paid, hosts


files = runtime_files()
assert len(files) > 30, f"scanned only {len(files)} files: is this the repo?"
paid, hosts = scan(files)
assert not paid, "paid LLM SDK / key / endpoint in runtime code:\n" + "\n".join(f"  {f}:{n}: {m}" for f, n, m in paid)
assert not hosts, ("web host not on the allowed list (a paid API? add it to HOSTS only if it is free and intended):\n"
                   + "\n".join(f"  {f}: {h}" for f, h in hosts))
# nothing paid is even installed in the venv the pipeline runs with
import importlib.util  # noqa: E402
installed = [m for m in ("anthropic", "openai") if importlib.util.find_spec(m)]
assert not installed, f"paid LLM SDK installed in {sys.executable}: {installed}"

# the scan catches what it should (a planted temp file, removed right after)
tmp = ROOT / "workspace" / "_paid_probe.py"
tmp.parent.mkdir(exist_ok=True)
tmp.write_text('import anthropic\nclient = anthropic.Anthropic()\nURL = "https://api.anthropic.com/v1/messages"\n'
               'x = "https://api.some-llm.example/v1"\ncoherent = "the edit felt coherent"\n', encoding="utf-8")
try:
    p, h = scan([tmp])
    assert {m for _, _, m in p} >= {"anthropic", "api.anthropic.com"} and not any(m == "cohere" for _, _, m in p), p
    assert ("workspace/_paid_probe.py", "api.some-llm.example") in h, h
finally:
    tmp.unlink()

print(f"no paid llm calls ok ({len(files)} runtime files, {len(HOSTS)} allowed hosts)")
