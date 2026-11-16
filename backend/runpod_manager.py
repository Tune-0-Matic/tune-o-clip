"""RunPod cloud GPU for tune-o-clip: one on-demand Pod (RTX 4000 Ada, fallback 3090 / 4090) that runs the same code with
the cloud profile (Whisper large-v3-turbo + qwen2.5:14b-instruct + NVENC), terminated when you switch it OFF.

  python backend/runpod_manager.py on [--gpu ID] [--cloud ALL|SECURE|COMMUNITY]
                                                 create + bootstrap the Pod (saved to workspace/runpod_state.json);
                                                 the panel passes the drawer's GPU + cloud type (else .env's)
  python backend/runpod_manager.py diagnose      no-cost checks: key, balance, the GPU's price/stock, create permission
  python backend/runpod_manager.py off           terminate it: compute and container disk are gone, $0.00 after
  python backend/runpod_manager.py status        refresh state; terminates a Pod past the idle / safety limits
  python backend/runpod_manager.py run -- ARGS   run `python ARGS` (clipper.py / queue_worker.py ...) on the Pod,
                                                 stream its JSON lines here, then pull new shorts back to output/
  python backend/runpod_manager.py set-key       read RUNPOD_API_KEY from stdin into .env (never on the command line)
  python backend/runpod_manager.py set-env K V   RUNPOD_GPU_ID / RUNPOD_CLOUD_TYPE into .env (the panel's drawer)
  python backend/runpod_manager.py gpus          the live GPU catalog (price, VRAM, availability) + account balance
  python backend/runpod_manager.py stats         the running Pod's GPU / CPU / RAM / disk (the panel's sidebar)
  python backend/runpod_manager.py watchdog      (on the Pod) terminate itself when idle / past the safety cap

Prints JSON lines like the other backend tools ({"event": "runpod", "stage": ..., "msg": ...}).
API: Pods REST v1 (https://rest.runpod.io/v1, Bearer key): POST /pods, GET /pods/{id}, DELETE /pods/{id} = terminate.
Credit protection: at most ONE tune-o-clip Pod (any leftover one is terminated before a new one is created; Pods
with other names are never touched), gpuCount 1 / volumeInGb 0 / containerDiskInGb <= 25 checked before every create,
a session spend cap (RUNPOD_MAX_SESSION_USD, default $1.50: elapsed s x $/hr / 3600) and an idle limit
(RUNPOD_IDLE_TIMEOUT_MIN, default 10 min after the last run). Both are enforced twice: by a watchdog ON the Pod (works
with this PC off or crashed) and by `status` from the panel every minute. No volume, so nothing is stored or billed
after terminate.
"""
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import twitch_api  # noqa: E402  (stdlib .env reader)

API = "https://rest.runpod.io/v1"
CATALOG = "https://api.runpod.io/v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud={cloud}"  # REST v2, read-only
GRAPHQL = "https://api.runpod.io/graphql"
CLOUDS = ("SECURE", "COMMUNITY")
SETTABLE = ("RUNPOD_GPU_ID", "RUNPOD_CLOUD_TYPE", "RUNPOD_IMAGE")  # non-secret .env keys a command line may set
GPU_IDS = ["NVIDIA RTX 4000 Ada Generation", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090"]
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"  # official, Ubuntu 24.04 + CUDA 12.8 (exists on Docker Hub)
CLOUD_MODEL, CLOUD_WHISPER = "qwen2.5:14b-instruct", "large-v3-turbo"
FALLBACK_MAX_USD_HR = float(os.environ.get("RUNPOD_FALLBACK_MAX_USD_HR") or 0.50)  # fallbacks never cost more
MIN_VRAM_GB = 16  # qwen2.5:14b Q4 (~9 GB + KV cache) + Whisper large-v3-turbo fp16 (~1.6 GB) + NVENC: ~13 GB
REMOTE = "/root/tune-o-clip"
POD_NAME = "tune-o-clip"  # the single-Pod lock only ever touches Pods with this name
RATE_FALLBACK = 0.70  # $/hr assumed for the spend cap when RunPod didn't report a price (above a 4090's)
STATE = ROOT / "workspace" / "runpod_state.json"
SSH_DIR = ROOT / "workspace" / "runpod_ssh"
BUSY, ACTIVITY = "/root/.toc_busy", "/root/.toc_activity"
STRIP = ("--whisper-model", "--ollama-model", "--device")  # the Pod's cloud defaults replace the local choices
PULL = ("output", "ready_to_post")


def emit(stage, msg, **kw):
    print(json.dumps({"event": "runpod", "stage": stage, "msg": msg, **kw}), flush=True)


def limits():
    """-> (idle minutes, max session USD) from .env / the Pod's env."""
    return float(os.environ.get("RUNPOD_IDLE_TIMEOUT_MIN") or 10), float(os.environ.get("RUNPOD_MAX_SESSION_USD") or 1.50)


def session_usd(now, created, rate):
    return max(0.0, now - created) * (rate or RATE_FALLBACK) / 3600


class RunPodError(RuntimeError):
    pass


class NoCapacity(RunPodError):
    pass


class Refused(RunPodError):
    """RunPod would not create this one Pod (the message says why when RunPod did)."""


def gql_literal(v, enums=(), key=None):
    """Python value -> GraphQL input literal (strings JSON-quoted, keys bare, the named enum fields unquoted)."""
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {gql_literal(x, enums, k)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(gql_literal(x, enums) for x in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    if key in enums:
        return str(v)
    return json.dumps(v) if isinstance(v, str) else str(v)


class RunPod:
    """Pods REST v1. opener is injectable for tests; errors never include the key."""

    RETRY = (429, 500, 502, 503, 504)  # rate limit / RunPod hiccup: wait and ask again, never "gone" or "failed"
    BACKOFF = (5, 10, 20, 30, 45, 60)  # s; ~3 min in all, covers RunPod's per-minute rate-limit window

    def __init__(self, key, opener=None, base=API, sleep=time.sleep):
        if not key:
            raise RunPodError("RUNPOD_API_KEY is not set: add it in the panel (RUNPOD KEY) or in .env")
        self.key, self.opener, self.base = key, opener or urllib.request.urlopen, base  # looked up now: tests block it
        self.sleep = sleep

    def call(self, method, path, body=None):
        """-> (HTTP code, parsed JSON / error text). path may be a full URL (the v2 catalog, GraphQL). 429 / 5xx are
        retried with backoff (live 2026-11-15: a 429 during the start poll was read as "Pod gone", and the cleanup
        terminate got a 429 too, leaving the Pod running until the next check)."""
        # POST is not idempotent: a 5xx may come after RunPod already made the Pod, so a create is only retried on
        # 429 (refused before any work); reads and DELETE retry on 429 and 5xx
        retry = self.RETRY if method in ("GET", "DELETE") else (429,)
        for wait in self.BACKOFF + (None,):
            code, data = self._call_once(method, path, body)
            if code not in retry or wait is None:
                return code, data
            emit("retry", f"RunPod HTTP {code} on {method} {path.split('?')[0][-40:]}: retrying in {wait} s")
            self.sleep(wait)

    def _call_once(self, method, path, body=None):
        req = urllib.request.Request(path if path.startswith("https://") else self.base + path, json.dumps(body).encode() if body is not None else None, method=method,
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                                              # Cloudflare in front of the API answers urllib's default agent with
                                              # 403 "error code: 1010" (a bot block, not a bad key)
                                              "User-Agent": "tune-o-clip/1.0"})
        try:
            with self.opener(req, timeout=60) as r:
                raw = r.read()
                return r.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")[:500].strip()
            return e.code, text or f"(empty body, request id {e.headers.get('x-request-id') if e.headers else '?'})"

    def create(self, body):
        code, data = self.call("POST", "/pods", body)
        if code == 403:  # REST refuses without a reason (empty body): sold out, but seen on in-stock GPUs too
            raise Refused(f"REST HTTP 403 {data}")
        if code >= 500 and "no instances" in str(data).lower():  # live: 500 "There are no instances currently available"
            raise Refused(f"REST HTTP {code}: {data}")
        if code not in (200, 201) or not isinstance(data, dict) or not data.get("id"):
            raise RunPodError(f"RunPod could not create the Pod (HTTP {code}): {data}")
        return data

    def deploy_graphql(self, body):
        """The same Pod through GraphQL podFindAndDeployOnDemand, which says why when it can't (errors[0].message,
        raised verbatim as Refused). A field this schema doesn't know is dropped and the call retried once."""
        fields = {"cloudType": body["cloudType"], "gpuCount": body["gpuCount"], "volumeInGb": body["volumeInGb"],
                  "containerDiskInGb": body["containerDiskInGb"], "gpuTypeId": body["gpuTypeIds"][0], "name": body["name"],
                  "imageName": body["imageName"], "dockerArgs": "", "ports": ",".join(body["ports"]),
                  "env": [{"key": k, "value": v} for k, v in body["env"].items()], "supportPublicIp": body.get("supportPublicIp", True)}
        for _ in range(2):
            code, data = self.call("POST", GRAPHQL, {"query": "mutation { podFindAndDeployOnDemand(input: %s) { id imageName machineId } }"
                                                              % gql_literal(fields, enums=("cloudType",))})
            errors = (data.get("errors") if isinstance(data, dict) else None) or []
            pod = ((data.get("data") or {}).get("podFindAndDeployOnDemand") if isinstance(data, dict) else None)
            if code == 200 and pod and pod.get("id") and not errors:
                return pod
            msg = str(errors[0].get("message")) if errors and isinstance(errors[0], dict) else f"HTTP {code}: {data}"
            unknown = next((f for f in fields if f'"{f}"' in msg and "not defined" in msg), None)
            if not unknown:
                raise Refused(f"GraphQL: {msg}")
            fields.pop(unknown)  # e.g. a field the GraphQL input type lacks: try once more without it
        raise Refused(f"GraphQL: {msg}")

    def catalog(self):
        """{gpu id: {name, memory, price{cloud}, avail{cloud}, vcpu{cloud}, ram{cloud}}} for both clouds (v2 catalog)."""
        out = {}
        for cloud in CLOUDS:
            code, data = self.call("GET", CATALOG.format(cloud=cloud))
            if code != 200 or not isinstance(data, dict):
                raise RunPodError(f"RunPod GPU catalog failed (HTTP {code}): {data}")
            for g in data.get("gpus") or []:
                e = out.setdefault(g["id"], {"id": g["id"], "name": g.get("name") or g["id"], "memory": g.get("memory"),
                                             "price": {}, "avail": {}, "vcpu": {}, "ram": {}})
                offered = g.get(cloud.lower(), True)  # the type exists in this cloud at all
                e["price"][cloud] = (g.get("price") or {}).get(cloud.lower()) if offered else None
                e["avail"][cloud] = (g.get("availability") or "NONE") if offered else "NONE"
                e["vcpu"][cloud], e["ram"][cloud] = g.get("minVcpuCount"), g.get("minRam")
        return out

    def balance(self):
        code, data = self.call("POST", GRAPHQL, {"query": "query { myself { clientBalance currentSpendPerHr } }"})
        me = ((data or {}).get("data") or {}).get("myself") if code == 200 and isinstance(data, dict) else None
        return me or {}

    def list(self):
        """Every Pod on the account (REST GET /pods; the list may come bare or wrapped)."""
        code, data = self.call("GET", "/pods")
        if code != 200:
            raise RunPodError(f"RunPod could not list Pods (HTTP {code}): {data}")
        pods = data.get("pods") or data.get("data") or [] if isinstance(data, dict) else data
        return [p for p in pods or [] if isinstance(p, dict)]

    def get(self, pod_id):
        """The Pod, or None when RunPod says it doesn't exist (404). Anything else (after retries) raises."""
        code, data = self.call("GET", f"/pods/{pod_id}")
        if code == 200 and isinstance(data, dict):
            return data
        if code in (400, 404):
            return None
        raise RunPodError(f"RunPod could not read Pod {pod_id} (HTTP {code}): {data}")

    def terminate(self, pod_id):
        code, data = self.call("DELETE", f"/pods/{pod_id}")
        if code not in (200, 204, 404):  # 404: already gone
            raise RunPodError(f"RunPod could not terminate {pod_id} (HTTP {code}): {data}")


# --- state --------------------------------------------------------------------------------------------------------

def load_state(path=STATE):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state, path=STATE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def clear_state(path=STATE):
    Path(path).unlink(missing_ok=True)


def is_ready(state=None):
    """True when runs should go to the Pod (the panel and the clipper ask this)."""
    return bool((state if state is not None else load_state()).get("ready"))


def should_terminate(now, created, last_activity, busy, idle_min, max_usd, rate):
    """-> why the Pod must go (str) or None. The spend cap always wins (even mid-run); idle only counts between runs."""
    spent = session_usd(now, created, rate)
    if spent >= max_usd:
        return f"spend cap: ${spent:.2f} of ${max_usd:.2f} this session"
    if not busy and now - last_activity > idle_min * 60:
        return f"idle {((now - last_activity) / 60):.0f} min (limit {idle_min:g} min)"
    return None


def ours(api):
    """Live tune-o-clip Pods on the account (the single-Pod lock and the orphan check use this)."""
    return [p for p in api.list() if p.get("name") == POD_NAME and p.get("desiredStatus") != "TERMINATED"]


def check_body(body):
    """The hard limits, checked on every create: one GPU, no volume, a small container disk."""
    if body.get("gpuCount") != 1 or body.get("volumeInGb") != 0 or not 0 < body.get("containerDiskInGb", 99) <= 25:
        raise RunPodError(f"refusing to create a Pod outside the limits: {({k: body.get(k) for k in ('gpuCount', 'volumeInGb', 'containerDiskInGb')})}")
    return body


# --- Pod lifecycle ------------------------------------------------------------------------------------------------

def pod_body(public_key, key, gpu_first=None, image=None, cloud="SECURE", gpus=None):
    idle_min, max_usd = limits()
    if gpus is None:
        gpus = [gpu_first] + [g for g in GPU_IDS if g != gpu_first] if gpu_first else list(GPU_IDS)
    return {"name": POD_NAME, "imageName": image or IMAGE, "gpuTypeIds": list(gpus), "gpuTypePriority": "custom",
            "gpuCount": 1, "cloudType": cloud,
            "supportPublicIp": True,  # SSH needs a public IP (always there on Secure Cloud; asked for on Community)
            "containerDiskInGb": 25, "volumeInGb": 0, "ports": ["22/tcp"], "interruptible": False,
            "env": {"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key, "NVIDIA_DRIVER_CAPABILITIES": "all",  # + NVENC
                    "TOC_CLOUD": "1", "TOC_RUNPOD_KEY": key,  # the watchdog terminates its own Pod with it
                    "RUNPOD_IDLE_TIMEOUT_MIN": f"{idle_min:g}", "RUNPOD_MAX_SESSION_USD": f"{max_usd:g}"}}


def cloud_type():
    v = (os.environ.get("RUNPOD_CLOUD_TYPE") or "ALL").upper()
    return v if v in ("ALL",) + CLOUDS else "ALL"


def placements(cat, selected=None, ctype="ALL", limit=5, fallback_max=None):
    """[(cloud, gpu id, $/hr)] to try, in order. FIRST the selected GPU in every allowed cloud that offers it, even
    when the catalog shows no stock (it lags; the user picked it). THEN fallbacks: any catalog GPU in stock with
    >= MIN_VRAM_GB, at most fallback_max $/hr (RUNPOD_FALLBACK_MAX_USD_HR, default 0.50), cheapest first. cat None
    (catalog down): the selected GPU, then the default preference list."""
    clouds = list(CLOUDS) if ctype == "ALL" else [ctype]
    selected = selected or GPU_IDS[0]  # no RUNPOD_GPU_ID: the RTX 4000 Ada is the choice
    fallback_max = FALLBACK_MAX_USD_HR if fallback_max is None else fallback_max
    if cat is None:
        return [(c, g, None) for g in dict.fromkeys([selected] + GPU_IDS) for c in clouds][:limit]
    sel = cat.get(selected) or {}
    if (sel.get("memory") or MIN_VRAM_GB) < MIN_VRAM_GB:
        raise RunPodError(f"{selected} has {sel.get('memory')} GB VRAM: the cloud models need {MIN_VRAM_GB} GB")
    first = [(c, selected, sel.get("price", {}).get(c)) for c in clouds if not sel or sel.get("price", {}).get(c) is not None]
    rest = sorted(((c, g["id"], g["price"].get(c)) for g in cat.values() if g["id"] != selected for c in clouds
                   if g["avail"].get(c) not in (None, "NONE") and g["price"].get(c) and g["price"][c] <= fallback_max
                   and (g.get("memory") or 0) >= MIN_VRAM_GB), key=lambda x: (x[2], x[1]))
    return (first + rest)[:limit]


def ensure_ssh_key(ssh_dir=SSH_DIR):
    key = Path(ssh_dir) / "id_ed25519"
    if not key.exists():
        key.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "tune-o-clip-runpod", "-f", str(key)], check=True)
    return key, key.with_suffix(".pub").read_text(encoding="utf-8").strip()


def ssh_cmd(state, remote, ssh_dir=SSH_DIR):
    return [shutil.which("ssh") or "ssh", "-i", str(Path(ssh_dir) / "id_ed25519"), "-p", str(state["port"]),
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={Path(ssh_dir) / 'known_hosts'}",
            "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=15", f"root@{state['ip']}", remote]


def wait_ready(api, state, sleep=time.sleep, tries=60):
    """Poll every 10 s until the Pod has its public IP and a mapped SSH port (~1-5 min while the image pulls)."""
    for i in range(tries):
        pod = api.get(state["pod_id"])
        if pod is None or pod.get("desiredStatus") == "TERMINATED":
            raise RunPodError("the Pod disappeared while starting (out of capacity or terminated)")
        port = (pod.get("portMappings") or {}).get("22")
        if pod.get("publicIp") and port:
            state.update(ip=pod["publicIp"], port=int(port))
            return state
        if i % 3 == 0:
            emit("deploying", f"Deploying Pod... ({pod.get('desiredStatus') or 'starting'}, {i * 10} s)")
        sleep(10)
    raise RunPodError("the Pod got no public IP / SSH port within 10 minutes")


def wait_ssh(state, run=subprocess.run, sleep=time.sleep, tries=36):
    for _ in range(tries):
        if run(ssh_cmd(state, "true"), capture_output=True).returncode == 0:
            return
        sleep(5)
    raise RunPodError("SSH to the Pod did not come up within 3 minutes")


def lf(data):
    """Bytes for Linux: CRLF / CR -> LF and no UTF-8 BOM, for text (no NUL byte in the first 8 KB); binary untouched.
    Every script, command input and uploaded text file goes through this: on Windows a text-mode pipe turns "\n"
    into "\r\n" and the working copy holds CRLF files, and bash on the Pod chokes on "\r"
    ("set: - : invalid option", "cd: $'/root/tune-o-clip\r'")."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    if b"\0" in data[:8192]:
        return data
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


# A Community-cloud Pod is someone else's machine: it gets only the Twitch app keys (clip metadata). Never the YouTube
# login cookies, the TikTok / YouTube app secrets or the RunPod key from .env (the watchdog gets its own via Pod env).
POD_ENV_KEYS = ("TWITCH_CLIENT_ID", "TWITCH_CLIENT_SECRET")
NEVER_UPLOAD = (".env", "cookies.txt")


def pod_env(raw):
    """.env bytes -> only the POD_ENV_KEYS lines (LF)."""
    lines = lf(raw).decode("utf-8", "replace").splitlines()
    return "".join(l + "\n" for l in lines if l.split("=", 1)[0].strip() in POD_ENV_KEYS).encode()


def upload_files(root=ROOT):
    """[(local path, name in the archive)]: the working copy's tracked files + the channel list. settings.json goes
    without its model choices (the Pod's cloud defaults apply) and .env only as pod_env() (see upload())."""
    # tracked AND new-but-not-ignored files: runpod_manager.py itself was still untracked on the first live run, so the
    # Pod had no watchdog script ("can't open file .../runpod_manager.py")
    files = subprocess.run(["git", "ls-files", "-z", "-co", "--exclude-standard"], cwd=root, capture_output=True,
                           text=True).stdout.split("\0")
    out = [(root / f, f) for f in files if f and f not in NEVER_UPLOAD and (root / f).is_file()]
    out += [(root / f, f) for f in ("workspace/twitch_channels.txt",) if (root / f).is_file()]
    return out


def upload(state, root=ROOT, run=subprocess.run):
    tar = Path(root) / "workspace" / "runpod_upload.tar"
    tar.parent.mkdir(parents=True, exist_ok=True)

    def add(t, name, raw, mode=0o644):
        info = tarfile.TarInfo(name)
        info.size, info.mode, info.mtime = len(raw), mode, int(time.time())
        t.addfile(info, io.BytesIO(raw))

    with tarfile.open(tar, "w") as t:
        for path, name in upload_files(Path(root)):
            add(t, name, lf(path.read_bytes()), 0o755 if name.endswith(".sh") else 0o644)  # LF on the Pod, always
        env = Path(root) / ".env"
        if env.is_file():
            add(t, ".env", pod_env(env.read_bytes()), 0o600)
        s = Path(root) / "workspace" / "settings.json"
        if s.exists():
            data = {k: v for k, v in json.loads(s.read_text(encoding="utf-8-sig")).items() if k not in ("whisper_model", "ollama_model")}
            add(t, "workspace/settings.json", json.dumps(data, indent=2).encode())
    with open(tar, "rb") as f:
        r = run(ssh_cmd(state, f"mkdir -p {REMOTE} && tar xf - -C {REMOTE}"), stdin=f, capture_output=True)
    tar.unlink(missing_ok=True)
    if r.returncode:
        raise RunPodError(f"upload to the Pod failed: {r.stderr.decode('utf-8', 'replace')[-300:]}")


BOOTSTRAP = r"""set -e
cd /root/tune-o-clip
say() { printf '{"event": "runpod", "stage": "%s", "msg": "%s"}\n' "$1" "$(printf '%s' "$2" | tr -d '"\\' | LC_ALL=C tr -c '\40-\176' ' ' | tail -c 400)"; }
# live 2026-11-15: a step died silently after the model pull; a failing step now names itself and the pull log's tail
trap 'say error "bootstrap step failed (exit $?): $BASH_COMMAND / $(tail -c 250 /root/pull.log 2>/dev/null) / $(df -h / | tail -1)"' ERR
export DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
# fast path: docker/Dockerfile.runpod baked ffmpeg, Ollama, the Python deps and both models in (RUNPOD_IMAGE);
# the generic base image still gets the full install below
BAKED=0
if [ -f /opt/toc_baked ] && command -v ffmpeg >/dev/null && command -v ollama >/dev/null; then BAKED=1; fi
if [ "$BAKED" = 1 ]; then
  export OLLAMA_MODELS=/opt/ollama/models HF_HOME=/opt/hf_cache  # SSH sessions don't get the image's ENV
  say bootstrap "Pre-baked image: starting Ollama..."
else
  say bootstrap "Installing ffmpeg + Ollama..."
  apt-get update -qq >/dev/null && apt-get install -y -qq ffmpeg curl zstd pciutils >/dev/null  # the Ollama installer needs zstd
  command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh >/dev/null 2>&1
fi
(OLLAMA_KEEP_ALIVE=10m nohup setsid ollama serve >/root/ollama.log 2>&1 < /dev/null &)
for i in $(seq 60); do curl -fs localhost:11434/api/tags >/dev/null && break; sleep 1; done
if [ "$BAKED" = 1 ] && ollama list | grep -q "^@MODEL@"; then
  say models "@MODEL@ + Whisper @WHISPER@ already in the image"
else
  say models "Loading @MODEL@ + Whisper @WHISPER@..."
  ollama pull @MODEL@ >/root/pull.log 2>&1  # its progress bars would bury a real error; the ERR trap shows its tail
fi
# a no-op in the baked image unless requirements.txt gained a package since the build
python3 -m pip install -q -r backend/requirements.txt nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
NV=$(python3 -c "import glob, os, site; print(':'.join(glob.glob(os.path.join(site.getsitepackages()[0], 'nvidia', '*', 'lib'))))")
printf 'export TOC_CLOUD=1 TOC_RUNPOD_RATE=@RATE@ TOC_RUNPOD_CREATED=@CREATED@ LD_LIBRARY_PATH=%s:${LD_LIBRARY_PATH:-}\n' "$NV" > /root/toc_env.sh
if [ "$BAKED" = 1 ]; then echo 'export OLLAMA_MODELS=/opt/ollama/models HF_HOME=/opt/hf_cache' >> /root/toc_env.sh; fi
# the Pod's own env (its id, the key the watchdog terminates with, the limits) is on PID 1 only: SSH sessions lack it
xargs -0 -n1 < /proc/1/environ | grep -E '^(RUNPOD_POD_ID|TOC_RUNPOD_KEY|RUNPOD_MAX_SESSION_USD|RUNPOD_IDLE_TIMEOUT_MIN)=' | sed 's/^/export /' >> /root/toc_env.sh
. /root/toc_env.sh
# warm-up only (the generic image: it also downloads Whisper there): a GPU CTranslate2 can't drive (live: an RTX 5080
# fallback) must not kill the bootstrap; runs fall back. The baked image has the model on disk: its first run loads it.
if [ "$BAKED" = 0 ]; then
  python3 -c "from faster_whisper import WhisperModel; WhisperModel('@WHISPER@', device='cuda', compute_type='float16')" >/dev/null 2>/root/whisper_warmup.log   || say warn "Whisper GPU warm-up failed on this GPU: runs use the CPU for Whisper (see /root/whisper_warmup.log)"
fi
# the encoder the renderer really gets here (some hosts block NVENC: "OpenEncodeSessionEx failed: unsupported device")
python3 -c "import sys; sys.path.insert(0, 'editor'); import renderer; print(renderer.detect_encoder()[0])" > /root/toc_encoder 2>/dev/null || true
touch /root/.toc_activity
(nohup setsid python3 backend/runpod_manager.py watchdog >/root/watchdog.log 2>&1 < /dev/null &)
say ready "Ready"
""".replace("@MODEL@", CLOUD_MODEL).replace("@WHISPER@", CLOUD_WHISPER)


def bootstrap_script(state):
    """The exact bytes of /tmp/toc_bootstrap.sh on the Pod: LF only (see lf())."""
    # @...@ placeholders: a bare "RATE" also matched inside the variable name TOC_RUNPOD_RATE
    return lf(BOOTSTRAP.replace("@RATE@", f"{state.get('cost_per_hr') or RATE_FALLBACK:g}")  # the watchdog's spend cap
              .replace("@CREATED@", f"{state.get('created') or 0:.0f}"))


BOOT_PATH = "/tmp/toc_bootstrap.sh"


def bootstrap(state, popen=subprocess.Popen, run=subprocess.run):
    """Copy the script to the Pod as a file (tar over SSH, binary), then run it with stdin closed. It used to be piped
    into `bash -s`: then the script WAS bash's stdin, and any step that reads stdin (an installer, apt, pip) could
    swallow the rest of the script, ending the bootstrap early with no error."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        data = bootstrap_script(state)
        info = tarfile.TarInfo(BOOT_PATH.rsplit("/", 1)[1])
        info.size, info.mode, info.mtime = len(data), 0o755, int(time.time())
        t.addfile(info, io.BytesIO(data))
    r = run(ssh_cmd(state, f"tar xf - -C {BOOT_PATH.rsplit('/', 1)[0]}"), input=buf.getvalue(), capture_output=True)
    if r.returncode:
        raise RunPodError(f"could not copy the bootstrap script to the Pod: {r.stderr.decode('utf-8', 'replace')[-300:]}")
    p = popen(ssh_cmd(state, f"bash {BOOT_PATH} </dev/null"), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
              stderr=subprocess.STDOUT)
    tail = []
    for raw in p.stdout:
        line = raw.decode("utf-8", "replace")
        if line.startswith('{"event"'):
            print(line.encode("ascii", "replace").decode(), end="", flush=True)  # live: a cp1252 pipe crashed on a pod line
        elif "█" not in line and "%" not in line[-30:]:  # skip download progress bars: keep the real error lines
            tail = (tail + [line.rstrip()])[-8:]
    if p.wait():
        raise RunPodError("bootstrap failed on the Pod: " + " | ".join(tail)[-500:])


def start_cloud_pod(api, key, gpu_first=None, state_path=STATE, ctype=None, ssh_ready=wait_ssh, do_upload=upload, do_bootstrap=bootstrap,
                    sleep=time.sleep, clock=time.time):
    state = load_state(state_path)
    live = ours(api)
    if state.get("ready") and any(p.get("id") == state.get("pod_id") for p in live):
        emit("ready", "Pod already on", gpu=state.get("gpu"), cost_per_hr=state.get("cost_per_hr"))
        return state
    for p in live:  # single-Pod lock: a leftover / half-started tune-o-clip Pod goes before a new one is made
        api.terminate(p["id"])
        emit("deploying", f"Terminated leftover Pod {p['id']} first (only one Pod at a time)")
    clear_state(state_path)
    _, public_key = ensure_ssh_key()
    try:
        cat = api.catalog()
    except RunPodError as e:
        emit("deploying", f"GPU catalog unavailable ({e}): trying the preferred GPUs blind")
        cat = None
    ctype = ctype or cloud_type()
    plan = placements(cat, gpu_first, ctype)
    if not plan:
        raise NoCapacity(f"nothing to try for {gpu_first or GPU_IDS[0]} ({ctype} cloud): pick another GPU or cloud type "
                         "in the RUNPOD drawer")
    pod = gpu = None
    tried = []
    for cloud, gid, price in plan:
        emit("deploying", f"Requesting {gid} ({cloud.title()} cloud{f', ${price:g}/hr' if price else ''})"
                          + (" [your choice]" if gid == plan[0][1] else " [fallback]") + "...")
        body = check_body(pod_body(public_key, key, image=os.environ.get("RUNPOD_IMAGE"), cloud=cloud, gpus=[gid]))
        try:
            pod = api.create(body)
        except Refused as rest_err:  # REST gives no reason: GraphQL deploys it or says why
            emit("deploying", f"{gid} / {cloud.lower()}: {rest_err}; asking GraphQL podFindAndDeployOnDemand...")
            try:
                pod = api.deploy_graphql(body)
            except Refused as e:
                tried.append(f"{gid} / {cloud.lower()}: {e}")
                emit("deploying", f"{gid} / {cloud.lower()} refused. {e}")
                continue
        gpu = gid
        if price and not pod.get("costPerHr"):
            pod["costPerHr"] = price  # the GraphQL answer has no price: the catalog's
        break
    if pod is None:
        raise NoCapacity("RunPod refused every option it tried:\n  " + "\n  ".join(tried)
                         + "\nPick another GPU or cloud type in the RUNPOD drawer, or run Test Deploy (no cost).")
    now = clock()
    state = {"pod_id": pod["id"], "gpu": gpu, "cost_per_hr": pod.get("costPerHr") or pod.get("adjustedCostPerHr"),
             "created": now, "last_activity": now, "busy": False, "ready": False}
    save_state(state, state_path)  # first, so OFF can always find (and terminate) it
    emit("deploying", f"Deploying Pod {pod['id']}...", gpu=gpu, cost_per_hr=state["cost_per_hr"])
    try:
        wait_ready(api, state, sleep)
        save_state(state, state_path)
        ssh_ready(state)
        emit("bootstrap", "Uploading tune-o-clip...")
        do_upload(state)
        do_bootstrap(state)
    except Exception as e:  # never leave a half-started Pod billing
        emit("error", f"Start failed ({e}): terminating the Pod")
        try:
            api.terminate(state["pod_id"])
        except RunPodError as t_err:  # keep the state: OFF / the next status (and the Pod's own guards) end it
            emit("error", f"COULD NOT TERMINATE Pod {state['pod_id']} ({t_err}): press RUNPOD GPU: OFF, or delete it in the "
                          "RunPod console")
            raise
        clear_state(state_path)
        raise
    state.update(ready=True, last_activity=clock())
    save_state(state, state_path)
    emit("ready", "Ready", gpu=state["gpu"], cost_per_hr=state["cost_per_hr"])
    return state


def terminate_cloud_pod(api, state_path=STATE, clock=time.time, why="switched OFF", everything=True):
    """Terminate this session's Pod and (everything=True) any other tune-o-clip Pod (orphans). -> ids terminated."""
    state = load_state(state_path)
    ids = [state["pod_id"]] if state.get("pod_id") else []
    if everything:
        ids += [p["id"] for p in ours(api) if p.get("id") not in ids]
    if not ids:
        clear_state(state_path)
        emit("off", "Local GPU (no Pod running): $0.00/hr")
        return []
    for pid in ids:
        api.terminate(pid)
    clear_state(state_path)
    cost = ""
    if state.get("created"):
        hours = (clock() - state["created"]) / 3600
        cost = f", ~${session_usd(clock(), state['created'], state.get('cost_per_hr')):.2f} for {hours:.2f} h"
    emit("off", f"Pod {', '.join(ids)} terminated ({why}{cost}). Billing stopped: $0.00/hr.")
    return ids


def status(api, state_path=STATE, clock=time.time):
    """Refresh: forget a Pod that no longer exists, terminate one past the limits. -> state ({} = off)."""
    state = load_state(state_path)
    idle_min, max_usd = limits()
    live = ours(api)
    if not state.get("pod_id"):
        if live:  # a Pod from a crashed / earlier session: the panel shows a red "click to terminate" banner
            rate = sum(p.get("costPerHr") or 0 for p in live) or None
            emit("orphan", f"WARNING: RunPod Pod Active (${rate or RATE_FALLBACK:.2f}/hr) - Click to Terminate",
                 pods=[p.get("id") for p in live], cost_per_hr=rate)
            return {"orphans": [p.get("id") for p in live]}
        emit("off", "Local GPU")
        return {}
    if not any(p.get("id") == state["pod_id"] for p in live):
        clear_state(state_path)
        emit("off", f"Pod {state['pod_id']} is gone (terminated by its watchdog or in the RunPod console): $0.00/hr")
        return {}
    now = clock()
    why = should_terminate(now, state["created"], state.get("last_activity", state["created"]), state.get("busy"),
                           idle_min, max_usd, state.get("cost_per_hr"))
    if why:
        terminate_cloud_pod(api, state_path, clock, why)
        return {}
    emit("ready" if state.get("ready") else "deploying", "Ready" if state.get("ready") else "Starting...",
         gpu=state.get("gpu"), cost_per_hr=state.get("cost_per_hr"), up_min=round((now - state["created"]) / 60),
         session_usd=round(session_usd(now, state["created"], state.get("cost_per_hr")), 2), max_usd=max_usd)
    return state


def diagnose(api, gpu=None, ctype=None):
    """No-cost checks (nothing is created): the key, balance + spend limit, the GPU's price / stock per cloud, and the
    REST create permission (gpuCount 0 is refused by the schema check, so a 400 proves the key may create Pods)."""
    gpu, ctype = gpu or os.environ.get("RUNPOD_GPU_ID") or GPU_IDS[0], ctype or cloud_type()
    code, _ = api.call("GET", "/pods")
    emit("diag", f"API key (GET /pods): {'OK' if code == 200 else f'HTTP {code}'}")
    me = api.call("POST", GRAPHQL, {"query": "query { myself { clientBalance currentSpendPerHr spendLimit } }"})[1]
    me = ((me or {}).get("data") or {}).get("myself") if isinstance(me, dict) else None
    emit("diag", "Account: " + (f"balance ${me.get('clientBalance')}, spending ${me.get('currentSpendPerHr')}/hr, "
                               f"spend limit {me.get('spendLimit')}" if me else "unreadable"))
    try:
        g = api.catalog().get(gpu)
    except RunPodError as e:
        g = None
        emit("diag", f"GPU catalog: {e}")
    emit("diag", f"{gpu}: " + (", ".join(f"{c.lower()} ${g['price'].get(c)}/hr {g['avail'].get(c)}" for c in CLOUDS)
                               + f", {g.get('memory')} GB VRAM" if g else "not in the catalog") + f" (cloud type {ctype})")
    body = dict(pod_body("ssh-ed25519 AAAA diagnose", "x", gpus=[gpu], cloud=CLOUDS[0] if ctype == "ALL" else ctype), gpuCount=0)
    code, data = api.call("POST", "/pods", body)
    emit("diag", "REST create permission: " + ("OK (the schema check answered, nothing created)" if code == 400 else
                                               f"HTTP {code} {str(data)[:200]}"))
    return code


# --- running a job on the Pod -------------------------------------------------------------------------------------

def remote_args(args, root=ROOT):
    """Local argv -> (Pod argv, [(local file, remote path)] to upload first). Paths under the repo map to REMOTE;
    other existing files (e.g. the panel's batch_urls.txt) are uploaded to workspace/_upload/; the model / device
    choices are dropped so the Pod's cloud profile applies."""
    root_n = os.path.normcase(str(Path(root).resolve()))
    out, uploads, it = [], [], iter(args)
    for a in it:
        if a in STRIP:
            next(it, None)
            continue
        p = Path(a)
        if p.is_absolute():
            full = os.path.normcase(str(p.resolve()))
            if full == root_n or full.startswith(root_n + os.sep):
                remote = REMOTE + "/" + p.resolve().relative_to(Path(root).resolve()).as_posix()
                if p.is_file():
                    uploads.append((p, remote))
                out.append(remote)
                continue
            if p.is_file():
                remote = f"{REMOTE}/workspace/_upload/{p.name}"
                uploads.append((p, remote))
                out.append(remote)
                continue
        out.append(a)
    return out, uploads


def rewrite_paths(folder, root=ROOT):
    """Pulled manifests / post.json hold the Pod's paths: point them at this machine's copy."""
    local = Path(root).resolve().as_posix()
    for f in Path(folder).rglob("*.json"):
        t = f.read_text(encoding="utf-8")
        if REMOTE in t:
            f.write_text(t.replace(REMOTE, local), encoding="utf-8")


def job_command(mapped, mark):
    """The Pod-side shell line for one run. BUSY pauses the watchdog's idle rule; a heartbeat also touches ACTIVITY
    every 30 s while this shell lives, so a long CPU render stays "active" even if another run's exit trap removed
    BUSY. The heartbeat ends by itself within 30 s of the shell (kill -0 $$) and holds no SSH fds, so a dropped
    session can't keep a Pod alive forever."""
    return (f". /root/toc_env.sh && cd {REMOTE} && touch {BUSY} {mark} && trap 'rm -f {BUSY}; touch {ACTIVITY}' EXIT || exit 1; "
            f"(while kill -0 $$ 2>/dev/null; do touch {ACTIVITY}; sleep 30; done) </dev/null >/dev/null 2>&1 & "
            f"python3 -u {shlex.join(mapped)}")


def run_remote(args, state_path=STATE, root=ROOT, popen=subprocess.Popen, run=subprocess.run, clock=time.time):
    state = load_state(state_path)
    if not state.get("ready"):
        raise RunPodError("the RunPod GPU is not ready: switch it ON first")
    mapped, uploads = remote_args(args, root)
    for local, remote in uploads:
        run(ssh_cmd(state, f"mkdir -p {shlex.quote(remote.rsplit('/', 1)[0])} && cat > {shlex.quote(remote)}"),
            input=lf(Path(local).read_bytes()), capture_output=True, check=True)
    mark = f"/tmp/toc_run_{int(clock())}"
    state.update(busy=True, last_activity=clock())
    save_state(state, state_path)
    emit("run", f"Running on the RunPod {state.get('gpu')}...")
    try:
        p = popen(ssh_cmd(state, job_command(mapped, mark)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                  errors="replace")
        for line in p.stdout:
            print(line, end="", flush=True)
        code = p.wait()
        pulled = pull_results(state, mark, root, popen)
        emit("pulled", f"{pulled} file(s) from the Pod in output/ and ready_to_post/")
        return code
    finally:
        state.update(busy=False, last_activity=clock())
        save_state(state, state_path)


def pull_results(state, mark, root=ROOT, popen=subprocess.Popen):
    """Everything the run wrote under output/ and ready_to_post/ (newer than its start mark) -> this machine."""
    cmd = (f"cd {REMOTE} && mkdir -p {' '.join(PULL)} && find {' '.join(PULL)} -type f -newer {mark} -print0 "
           f"| tar cf - --null -T -")
    p = popen(ssh_cmd(state, cmd), stdout=subprocess.PIPE)
    n = 0
    try:
        with tarfile.open(fileobj=p.stdout, mode="r|") as t:
            for m in t:
                if m.isfile():
                    t.extract(m, Path(root), filter="data")  # "data": no absolute paths, no .. escapes
                    n += 1
    except tarfile.ReadError:  # nothing new: an empty stream
        pass
    p.wait()
    for d in PULL:
        if (Path(root) / d).exists():
            rewrite_paths(Path(root) / d, root)
    return n


POD_STATS = r"""
import json, os, shutil, subprocess, time
def cg(name):
    try:
        return open("/sys/fs/cgroup/" + name).read().strip()
    except OSError:
        return None
def usage():  # CPU seconds used by this container: cgroup v2 cpu.stat, else v1 cpuacct (RunPod hosts have both kinds)
    s = cg("cpu.stat")
    if s and "usage_usec" in s:
        return int(s.split("usage_usec")[1].split()[0]) / 1e6
    v = cg("cpuacct/cpuacct.usage") or cg("cpu,cpuacct/cpuacct.usage")
    return int(v) / 1e9 if v and v.isdigit() else None
q = (cg("cpu.max") or "max").split()
quota, period = cg("cpu/cpu.cfs_quota_us"), cg("cpu/cpu.cfs_period_us")
if q[0] != "max":
    vcpu = int(q[0]) / int(q[1])
elif quota and period and quota.lstrip("-").isdigit() and int(quota) > 0:
    vcpu = int(quota) / int(period)  # v1 quota
else:
    vcpu = os.cpu_count()
u1, t1 = usage(), time.time(); time.sleep(0.5); u2, t2 = usage(), time.time()
cpu = min(100.0, 100 * (u2 - u1) / (t2 - t1) / vcpu) if u1 is not None and u2 is not None else None
mem = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in open("/proc/meminfo")}
mmax, mcur = cg("memory.max"), cg("memory.current")
lim, cur = cg("memory/memory.limit_in_bytes"), cg("memory/memory.usage_in_bytes")  # v1
if mmax and mmax.isdigit():
    total, used = int(mmax), int(mcur or 0)
elif lim and lim.isdigit() and int(lim) < 1 << 50:  # v1 limit (a huge number = unlimited)
    st = dict(l.split() for l in (cg("memory/memory.stat") or "").splitlines() if len(l.split()) == 2)
    total, used = int(lim), int(cur or 0) - int(st.get("total_inactive_file", 0))  # minus reclaimable page cache
else:
    total, used = mem["MemTotal"], mem["MemTotal"] - mem["MemAvailable"]
g = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().split(", ")
try:
    encoder = open("/root/toc_encoder").read().strip()
except OSError:
    encoder = None
print(json.dumps({"cpu": cpu, "vcpu": round(vcpu, 1), "ram_used": used / 1e9, "ram_total": total / 1e9, "encoder": encoder,
                  "disk_free": shutil.disk_usage("/").free / 1e9,
                  "gpu": {"name": g[0], "util": float(g[1]), "used": float(g[2]), "total": float(g[3]), "temp": float(g[4])} if len(g) == 5 else {}}))
"""


def pod_stats(state_path=STATE, run=subprocess.run):
    """One sample of the running Pod's hardware over SSH -> dict (empty when the Pod isn't ready / unreachable)."""
    state = load_state(state_path)
    if not state.get("ready"):
        return {}
    r = run(ssh_cmd(state, "python3 -"), input=lf(POD_STATS), capture_output=True, timeout=30)  # bytes: no \r added
    try:
        out = r.stdout.decode("utf-8", "replace") if isinstance(r.stdout, bytes) else r.stdout
        d = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {}
    return dict(d, gpu_type=state.get("gpu"), cost_per_hr=state.get("cost_per_hr"))


# --- on the Pod -----------------------------------------------------------------------------------------------------

def watchdog(api, pod_id, clock=time.time, sleep=time.sleep, busy=BUSY, activity=ACTIVITY, rate=None, created=None):
    """On the Pod: the same rules as status(), counted from the Pod's creation time (TOC_RUNPOD_CREATED, written by the
    bootstrap; now if missing)."""
    created = created if created is not None else float(os.environ.get("TOC_RUNPOD_CREATED") or 0) or clock()
    idle_min, max_usd = limits()
    rate = rate or float(os.environ.get("TOC_RUNPOD_RATE") or RATE_FALLBACK)
    while True:
        try:
            last = os.path.getmtime(activity)
        except OSError:
            last = created
        why = should_terminate(clock(), created, last, os.path.exists(busy), idle_min, max_usd, rate)
        if why:
            print(f"terminating {pod_id}: {why}", flush=True)
            api.terminate(pod_id)
            return why
        sleep(60)


# --- .env -----------------------------------------------------------------------------------------------------------

def set_env_key(name, value, path=ROOT / ".env"):
    """Set NAME=value in .env, keeping every other line (and a Notepad BOM) as it is."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    bom = path.exists() and path.read_bytes().startswith(b"\xef\xbb\xbf")
    lines = [l for l in text.splitlines() if not l.strip().startswith(name + "=")]
    lines.append(f"{name}={value}")
    # atomic: write a temp file, then swap it in. An ON started right after a drawer click reads .env meanwhile and
    # must never see it truncated (that read lost RUNPOD_API_KEY).
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(("﻿" if bom else "") + "\n".join(lines) + "\n", encoding="utf-8")
    for wait in (0.05, 0.1, 0.2, 0.4, None):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: another process has .env open for a moment
            if wait is None:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(wait)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv else ""
    twitch_api.load_env()
    try:
        if mode == "set-key":
            key = sys.stdin.readline().strip()
            if not key:
                raise RunPodError("no key given")
            set_env_key("RUNPOD_API_KEY", key)
            emit("key", "RunPod API key saved to .env")
            return 0
        if mode == "set-env":
            pairs = argv[1:]  # KEY VALUE [KEY VALUE ...]: the drawer saves GPU + cloud type in one process (no lost update)
            if not pairs or len(pairs) % 2 or any(k not in SETTABLE for k in pairs[::2]) \
                    or any(not v.strip() or "\r" in v or "\n" in v for v in pairs[1::2]):
                raise RunPodError(f"set-env takes {', '.join(SETTABLE)} each with a one-line value")
            for k, v in zip(pairs[::2], pairs[1::2]):  # all validated before anything is written
                set_env_key(k, v.strip())
                emit("env", f"{k} = {v.strip()} saved to .env")
            return 0
        if mode == "watchdog":
            watchdog(RunPod(os.environ.get("TOC_RUNPOD_KEY")), os.environ["RUNPOD_POD_ID"])
            return 0
        if mode == "stats":
            print(json.dumps({"event": "runpod_stats", **pod_stats()}), flush=True)
            return 0
        if mode == "status" and not os.environ.get("RUNPOD_API_KEY"):
            emit("off", "Local GPU (no RunPod key)")
            return 0
        api = RunPod(os.environ.get("RUNPOD_API_KEY"))
        opt = lambda name: argv[argv.index(name) + 1] if name in argv[:-1] else None
        if mode == "on":  # the panel passes the drawer's choice: no race with its background .env write
            ctype = (opt("--cloud") or "").upper()
            start_cloud_pod(api, api.key, opt("--gpu") or os.environ.get("RUNPOD_GPU_ID"),
                            ctype=ctype if ctype in ("ALL",) + CLOUDS else None)
        elif mode == "diagnose":
            diagnose(api, opt("--gpu"), (opt("--cloud") or "").upper() or None)
        elif mode == "off":
            terminate_cloud_pod(api)
        elif mode == "status":
            status(api)
        elif mode == "gpus":
            cat = api.catalog()
            idle_min, max_usd = limits()
            ct = cloud_type()
            clouds = list(CLOUDS) if ct == "ALL" else [ct]
            best = lambda g: min([g["price"][c] for c in clouds if g["avail"].get(c) not in (None, "NONE") and g["price"].get(c)] or [99])
            gpus = sorted(cat.values(), key=lambda g: (best(g) == 99, best(g), g["name"]))
            print(json.dumps({"event": "runpod_gpus", "gpus": gpus, "balance": api.balance().get("clientBalance"),
                              "min_vram_gb": MIN_VRAM_GB,
                              "selected": os.environ.get("RUNPOD_GPU_ID") or GPU_IDS[0], "cloud_type": ct,
                              "max_usd": max_usd, "idle_min": idle_min}), flush=True)
        elif mode == "run":
            return run_remote(argv[2:] if argv[1:2] == ["--"] else argv[1:])
        else:
            print(__doc__)
            return 2
        return 0
    except (RunPodError, OSError, subprocess.CalledProcessError) as e:
        print(json.dumps({"event": "error", "msg": f"[RUNPOD] {e}"}), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
