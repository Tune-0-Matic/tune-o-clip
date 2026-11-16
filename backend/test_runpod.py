"""RunPod cloud GPU tests (no network, no money): the Pods REST calls are mocked. Create -> state saved first ->
ready -> terminate; a failed start terminates the Pod; idle / safety-cap guards; argv mapping for runs; pulled paths;
the .env key; and the profile switch (TOC_CLOUD=1 -> large-v3-turbo + qwen2.5:14b-instruct, else the local defaults).
Run: python backend/test_runpod.py
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Mock-only: the real RUNPOD_API_KEY must never be used here. An empty value also stops .env from supplying it
# (load_env keeps existing variables), and every real HTTP request fails the test.
os.environ["RUNPOD_API_KEY"] = ""
for k in [k for k in os.environ if k.startswith(("TOC_", "RUNPOD_")) and k != "RUNPOD_API_KEY"]:
    del os.environ[k]
import urllib.request  # noqa: E402
blocked = []
def no_network(req, *a, **k):
    blocked.append(getattr(req, "full_url", req))
    raise AssertionError(f"real network call in a test: {getattr(req, 'full_url', req)}")
urllib.request.urlopen = no_network
import runpod_manager as rp  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="runpod_test_"))
state_path = tmp / "runpod_state.json"
quiet = lambda: contextlib.redirect_stdout(io.StringIO())


class Resp(io.BytesIO):
    def __init__(self, status, data=b""):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# the v2 GPU catalog as RunPod returns it (per cloud); availability NONE = sold out
CATALOG = {
    "SECURE": [{"id": "NVIDIA A40", "name": "A40", "memory": 48, "secure": True, "community": False,
                "price": {"secure": 0.59, "community": None}, "availability": "NONE"},
               {"id": "NVIDIA RTX 4000 Ada Generation", "name": "RTX 4000 Ada", "memory": 20, "secure": True, "community": True,
                "price": {"secure": 0.28, "community": 0.2}, "availability": "LOW", "minVcpuCount": 9, "minRam": 50},
               {"id": "NVIDIA GeForce RTX 3090", "name": "RTX 3090", "memory": 24, "secure": True, "community": True,
                "price": {"secure": 0.5, "community": 0.22}, "availability": "NONE"},
               {"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "memory": 24, "secure": True, "community": True,
                "price": {"secure": 0.89, "community": 0.34}, "availability": "LOW", "minVcpuCount": 16, "minRam": 61}],
    "COMMUNITY": [{"id": "NVIDIA A40", "name": "A40", "memory": 48, "secure": True, "community": False,
                   "price": {"secure": 0.59, "community": None}, "availability": "NONE"},
                  {"id": "NVIDIA RTX 4000 Ada Generation", "name": "RTX 4000 Ada", "memory": 20, "secure": True, "community": True,
                   "price": {"secure": 0.28, "community": 0.2}, "availability": "HIGH", "minVcpuCount": 6, "minRam": 31},
                  {"id": "NVIDIA GeForce RTX 3090", "name": "RTX 3090", "memory": 24, "secure": True, "community": True,
                   "price": {"secure": 0.5, "community": 0.22}, "availability": "LOW", "minVcpuCount": 8, "minRam": 30},
                  {"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "memory": 24, "secure": True, "community": True,
                   "price": {"secure": 0.89, "community": 0.34}, "availability": "NONE"}]}


class FakeRunPod:
    """A tiny RunPod: REST v2 pods (POST / GET / DELETE), the v2 GPU catalog and GraphQL myself. A create for a GPU in
    `full` (as (gpu, cloud)) answers 400 with a human-readable detail, like v2 with no capacity."""

    NO_CAP = "There are no instances currently available with the requested specifications."

    def __init__(self, ready_after=1, full=()):
        self.calls, self.pods, self.ready_after, self.gets, self.full = [], {}, ready_after, 0, set(full)

    def __call__(self, req, timeout=None):
        body = json.loads(req.data) if req.data else None
        assert req.headers["Authorization"] == "Bearer KEY123" and req.get_header("User-agent") == "tune-o-clip/1.0"
        self.calls.append((req.get_method(), req.full_url.replace(rp.API, ""), body))
        method, path = req.get_method(), req.full_url.replace(rp.API, "")
        if req.full_url.startswith("https://api.runpod.io/v2/catalog/gpus"):
            cloud = path.split("cloud=")[1]
            return Resp(200, json.dumps({"gpus": CATALOG[cloud]}).encode())
        if path == rp.GRAPHQL:
            return Resp(200, json.dumps({"data": {"myself": {"clientBalance": 9.73, "currentSpendPerHr": 0, "spendLimit": 80}}}).encode())
        if method == "POST" and path == "/pods":
            assert set(body) <= {"name", "image", "gpu", "cloud", "disk", "ports", "env"}, f"v2 rejects unknown fields: {sorted(body)}"
            if body["gpu"]["count"] == 0:
                raise urllib.error.HTTPError(req.full_url, 422, "Unprocessable", {}, io.BytesIO(
                    b'{"detail":"Request validation failed.","errors":["$.gpu.count: minimum: got 0, want 1"],"status":422}'))
            if (body["gpu"]["id"], body["cloud"]) in self.full:
                raise urllib.error.HTTPError(req.full_url, 400, "Bad", {}, io.BytesIO(json.dumps({"detail": self.NO_CAP, "status": 400}).encode()))
            self.pods["pod1"] = {"id": "pod1", "name": body["name"], "status": "PROVISIONING", "cost": 0.28, "ssh": None}
            return Resp(201, json.dumps(self.pods["pod1"]).encode())
        if method == "GET" and path == "/pods":
            return Resp(200, json.dumps({"pods": list(self.pods.values()), "pagination": {"hasNextPage": False, "nextCursor": None}}).encode())
        pid = path.rsplit("/", 1)[1]
        if method == "GET":
            if pid not in self.pods:
                raise urllib.error.HTTPError(req.full_url, 404, "Pod not found.", {}, io.BytesIO(b"not found"))
            self.gets += 1
            if self.gets > self.ready_after:
                self.pods[pid].update(status="RUNNING", ssh={"direct": {"host": "203.0.113.7", "port": 10341, "username": "root"}})
            return Resp(200, json.dumps(self.pods[pid]).encode())
        if method == "DELETE":
            self.pods.pop(pid, None)
            return Resp(204)
        raise AssertionError((method, path))


rp.ensure_ssh_key = lambda ssh_dir=None: (tmp / "key", "ssh-ed25519 AAAATEST tune-o-clip")
fake = FakeRunPod()
api = rp.RunPod("KEY123", opener=fake)
done = []
with quiet() as out:
    state = rp.start_cloud_pod(api, "KEY123", None, state_path, ssh_ready=lambda s: done.append("ssh"),
                               do_upload=lambda s: done.append("upload"), do_bootstrap=lambda s: done.append("boot"),
                               sleep=lambda s: None, clock=lambda: 1000.0)
assert fake.calls[0][:2] == ("GET", "/pods"), fake.calls[0]  # single-Pod lock: the account is checked first
assert [c[1].split("cloud=")[-1] for c in fake.calls[1:3]] == ["SECURE", "COMMUNITY"], "live catalog before creating"
posts = [c for c in fake.calls if c[:2] == ("POST", "/pods")]
assert len(posts) == 1, posts
body = posts[0][2]
# the default GPU (RTX 4000 Ada) has capacity in Secure: that's the first try, one type per attempt
assert body["gpu"] == {"id": "NVIDIA RTX 4000 Ada Generation", "count": 1} and body["cloud"] == "SECURE"
assert "mounts" not in body and body["disk"] == 25 and body["ports"] == ["22/tcp"]
assert body["env"]["PUBLIC_KEY"].startswith("ssh-ed25519") and body["env"]["TOC_CLOUD"] == "1" and body["env"]["TOC_RUNPOD_KEY"] == "KEY123"
assert body["image"] == rp.IMAGE and "ubuntu2404" in rp.IMAGE
assert done == ["ssh", "upload", "boot"] and state["ready"] and state["ip"] == "203.0.113.7" and state["port"] == 10341
saved = json.loads(state_path.read_text())
assert saved["pod_id"] == "pod1" and saved["cost_per_hr"] == 0.28 and saved["ready"], saved
assert rp.is_ready(saved) and '"stage": "ready"' in out.getvalue()
assert "KEY123" not in out.getvalue(), "the key is never printed"
# placements: the selected GPU first (each cloud where it's free), then the cheapest free fallbacks; sold-out types skipped
cat = rp.RunPod("KEY123", opener=FakeRunPod()).catalog()
assert cat["NVIDIA RTX 4000 Ada Generation"]["avail"] == {"SECURE": "LOW", "COMMUNITY": "HIGH"}
assert cat["NVIDIA RTX 4000 Ada Generation"]["vcpu"] == {"SECURE": 9, "COMMUNITY": 6} and cat["NVIDIA GeForce RTX 3090"]["memory"] == 24
assert rp.placements(cat, None, "ALL") == [("SECURE", "NVIDIA RTX 4000 Ada Generation", 0.28),  # no choice: the Ada,
                                          ("COMMUNITY", "NVIDIA RTX 4000 Ada Generation", 0.2),  # both clouds,
                                          ("COMMUNITY", "NVIDIA GeForce RTX 3090", 0.22)]  # then <= $0.50 in stock
plan = rp.placements(cat, "NVIDIA A40", "ALL")
assert plan[0] == ("SECURE", "NVIDIA A40", 0.59), plan  # the drawer's A40 first, though the catalog says sold out
assert all(g != "NVIDIA A40" for _, g, _ in plan[1:]) and [p for _, _, p in plan[1:]] == sorted(p for _, _, p in plan[1:]), plan
assert all(p <= 0.50 for _, _, p in plan[1:]) and ("SECURE", "NVIDIA GeForce RTX 4090", 0.89) not in plan, "fallbacks <= $0.50/hr"
assert rp.placements(cat, "NVIDIA A40", "COMMUNITY")[0][1] != "NVIDIA A40", "A40 isn't offered on Community: not tried there"
assert rp.placements(cat, "NVIDIA A40", "ALL", fallback_max=0.21) == [("SECURE", "NVIDIA A40", 0.59), ("COMMUNITY", "NVIDIA RTX 4000 Ada Generation", 0.2)]
assert rp.placements(None, None, "SECURE")[0] == ("SECURE", "NVIDIA RTX 4000 Ada Generation", None)  # catalog down: blind
small = {"NVIDIA GeForce RTX 3070": {"id": "NVIDIA GeForce RTX 3070", "memory": 8, "price": {"COMMUNITY": 0.13}, "avail": {"COMMUNITY": "HIGH"}}, **cat}
assert all(g != "NVIDIA GeForce RTX 3070" for _, g, _ in rp.placements(small, None, "ALL")), "8 GB never a fallback"
try:
    rp.placements(small, "NVIDIA GeForce RTX 3070", "ALL")
    raise AssertionError("an 8 GB choice must be refused")
except rp.RunPodError as e:
    assert "need 16 GB" in str(e)
assert rp.pod_body("k", "x", "NVIDIA GeForce RTX 4090")["gpu"] == {"id": "NVIDIA GeForce RTX 4090", "count": 1}
assert rp.pod_body("k", "x")["gpu"]["id"] == rp.GPU_IDS[0]

# the drawer's A40: v2 400 (no capacity, detail printed verbatim) -> the next placement deploys
every = {(g["id"], c) for c in rp.CLOUDS for g in CATALOG[c]}
fake_a40 = FakeRunPod(full={("NVIDIA A40", "SECURE")})
with quiet() as out:
    st = rp.start_cloud_pod(rp.RunPod("KEY123", opener=fake_a40), "KEY123", "NVIDIA A40", tmp / "a40_state.json", ctype="ALL",
                            ssh_ready=lambda s: None, do_upload=lambda s: None, do_bootstrap=lambda s: None, sleep=lambda s: None)
rest_posts = [c[2] for c in fake_a40.calls if c[:2] == ("POST", "/pods")]
assert rest_posts[0]["gpu"]["id"] == "NVIDIA A40" and rest_posts[0]["cloud"] == "SECURE", "the selection goes first"
log = out.getvalue()
assert f"HTTP 400: {FakeRunPod.NO_CAP}" in log and "[your choice]" in log, log
assert st["pod_id"] == "pod1" and st["gpu"] != "NVIDIA A40" and st["cost_per_hr"] == 0.28, st
# everything refused: one error listing every reason, nothing created or saved
fake_none = FakeRunPod(full=every)
with quiet() as out:
    try:
        rp.start_cloud_pod(rp.RunPod("KEY123", opener=fake_none), "KEY123", None, tmp / "none_state.json", ssh_ready=None,
                           do_upload=None, do_bootstrap=None, sleep=lambda s: None)
        raise AssertionError("should raise")
    except rp.NoCapacity as e:
        assert "refused every option" in str(e) and FakeRunPod.NO_CAP in str(e) and "Test Deploy" in str(e), str(e)
assert not fake_none.pods and not (tmp / "none_state.json").exists(), "nothing created, nothing saved"
# 402 (no balance) stops at once: no other GPU is tried
class Broke(FakeRunPod):
    def __call__(self, req, timeout=None):
        if req.get_method() == "POST" and req.full_url.endswith("/pods"):
            self.calls.append(("POST", "/pods", json.loads(req.data)))
            raise urllib.error.HTTPError(req.full_url, 402, "x", {}, io.BytesIO(b'{"detail":"Insufficient balance.","status":402}'))
        return super().__call__(req, timeout)
broke = Broke()
with quiet():
    try:
        rp.start_cloud_pod(rp.RunPod("KEY123", opener=broke), "KEY123", None, tmp / "broke.json", ssh_ready=None,
                           do_upload=None, do_bootstrap=None, sleep=lambda s: None)
        raise AssertionError("should raise")
    except rp.NoCapacity:
        raise AssertionError("402 is not a capacity problem")
    except rp.RunPodError as e:
        assert "HTTP 402" in str(e) and "Insufficient balance" in str(e), str(e)
assert sum(c[:2] == ("POST", "/pods") for c in broke.calls) == 1, "402: no further GPUs tried"
# main(): `on --gpu X --cloud Y` reaches start_cloud_pod as the first choice (no race with the .env write)
got = {}
real_start = rp.start_cloud_pod
rp.start_cloud_pod = lambda api, key, gpu=None, state_path=None, ctype=None, **k: got.update(gpu=gpu, ctype=ctype)
os.environ["RUNPOD_API_KEY"] = "KEY123"
os.environ["RUNPOD_GPU_ID"] = "NVIDIA GeForce RTX 3090"  # a stale .env value
with quiet():
    assert rp.main(["on", "--gpu", "NVIDIA A40", "--cloud", "secure"]) == 0
assert got == {"gpu": "NVIDIA A40", "ctype": "SECURE"}, got
with quiet():
    rp.main(["on"])
assert got == {"gpu": "NVIDIA GeForce RTX 3090", "ctype": None}, got  # no flags: .env's choice
rp.start_cloud_pod = real_start
del os.environ["RUNPOD_GPU_ID"]
os.environ["RUNPOD_API_KEY"] = ""
# diagnose: no Pod is ever created (only the schema-rejected gpuCount 0 probe)
fake_d = FakeRunPod()
with quiet() as out:
    assert rp.diagnose(rp.RunPod("KEY123", opener=fake_d), "NVIDIA A40", "ALL") == 422
d = out.getvalue()
assert "API key (GET /pods): OK" in d and "balance $9.73" in d and "spend limit 80" in d and "NVIDIA A40: secure $0.59/hr NONE" in d, d
assert "REST create permission: OK" in d and not fake_d.pods, "diagnose never deploys"
os.environ["RUNPOD_CLOUD_TYPE"] = "COMMUNITY"
assert rp.cloud_type() == "COMMUNITY"
os.environ["RUNPOD_CLOUD_TYPE"] = "bogus"
assert rp.cloud_type() == "ALL"
del os.environ["RUNPOD_CLOUD_TYPE"]
assert rp.RunPod("KEY123", opener=FakeRunPod()).balance()["clientBalance"] == 9.73

# ON again while it's ready: reuse, no second Pod
n = len(fake.calls)
with quiet():
    rp.start_cloud_pod(api, "KEY123", None, state_path, ssh_ready=None, do_upload=None, do_bootstrap=None)
assert [c[:2] for c in fake.calls[n:]] == [("GET", "/pods")], fake.calls[n:]  # listed, reused, nothing created

# OFF: DELETE, state gone, $0.00
with quiet() as out:
    assert rp.terminate_cloud_pod(api, state_path, clock=lambda: 1000 + 3600) == ["pod1"]
assert fake.calls[-1][:2] == ("DELETE", "/pods/pod1") and not state_path.exists()
assert "$0.00/hr" in out.getvalue() and "~$0.28 for 1.00 h" in out.getvalue(), out.getvalue()
with quiet() as out:
    assert rp.terminate_cloud_pod(api, state_path) == [] and "no Pod running" in out.getvalue()

# a start that fails half-way never leaves a Pod billing
fake2 = FakeRunPod()
api2 = rp.RunPod("KEY123", opener=fake2)
def broken(s):
    raise rp.RunPodError("pip exploded")
with quiet() as out:
    try:
        rp.start_cloud_pod(api2, "KEY123", None, state_path, ssh_ready=lambda s: None, do_upload=lambda s: None,
                           do_bootstrap=broken, sleep=lambda s: None)
        raise AssertionError("should raise")
    except rp.RunPodError:
        pass
assert fake2.calls[-1][:2] == ("DELETE", "/pods/pod1") and not fake2.pods and not state_path.exists()
assert "terminating the Pod" in out.getvalue()
# a Pod left from an unfinished start is replaced, not trusted
rp.save_state({"pod_id": "old", "ready": False, "created": 0}, state_path)
fake3 = FakeRunPod()
fake3.pods["old"] = {"id": "old", "name": "tune-o-clip", "status": "RUNNING"}
fake3.pods["mine"] = {"id": "mine", "name": "my-stable-diffusion", "status": "RUNNING"}  # not ours: untouched
with quiet():
    rp.start_cloud_pod(rp.RunPod("KEY123", opener=fake3), "KEY123", None, state_path, ssh_ready=lambda s: None,
                       do_upload=lambda s: None, do_bootstrap=lambda s: None, sleep=lambda s: None)
assert ("DELETE", "/pods/old", None) in fake3.calls and json.loads(state_path.read_text())["pod_id"] == "pod1"
assert ("DELETE", "/pods/mine", None) not in fake3.calls and "mine" in fake3.pods, "other Pods are never touched"
assert sum(1 for p in fake3.pods.values() if p.get("name") == "tune-o-clip") == 1, "single-Pod lock"
state_path.unlink()
# hard limits on every create
for bad in ({"gpu": {"id": "x", "count": 2}}, {"mounts": {"persistent": {"size": 20, "path": "/workspace"}}}, {"disk": 50}):
    try:
        rp.check_body({**rp.pod_body("k", "x"), **bad})
        raise AssertionError(f"should refuse {bad}")
    except rp.RunPodError as e:
        assert "refusing" in str(e)
assert rp.check_body(rp.pod_body("k", "x"))["env"]["RUNPOD_MAX_SESSION_USD"] == "1.5"
assert rp.pod_body("k", "x")["env"]["RUNPOD_IDLE_TIMEOUT_MIN"] == "10"

# no key: a clear message, no request
try:
    rp.RunPod("")
    raise AssertionError("should raise")
except rp.RunPodError as e:
    assert "RUNPOD_API_KEY" in str(e)

# --- guards: idle (only while not busy) and the safety cap (always) ---------------------------------------------------
assert rp.limits() == (10, 1.5)  # defaults: 10 min idle, $1.50 per session
assert rp.should_terminate(1000 + 11 * 60, 1000, 1000, False, 10, 1.5, 0.28).startswith("idle 11 min")
assert rp.should_terminate(1000 + 11 * 60, 1000, 1000, True, 10, 1.5, 0.28) is None  # a long run is not idle
assert rp.should_terminate(1000 + 9 * 60, 1000, 1000, False, 10, 1.5, 0.28) is None
cap_s = 1.5 / 0.28 * 3600  # ~5.36 h on an RTX 4000 Ada
assert rp.should_terminate(1000 + cap_s - 60, 1000, 1000 + cap_s - 60, True, 10, 1.5, 0.28) is None
assert rp.should_terminate(1000 + cap_s + 1, 1000, 1000 + cap_s, True, 10, 1.5, 0.28).startswith("spend cap: $1.50 of $1.50")
assert rp.should_terminate(1000 + 2.2 * 3600, 1000, 1000 + 2.2 * 3600, True, 10, 1.5, None).startswith("spend cap")  # no price: $0.70/hr
assert abs(rp.session_usd(3600, 0, 0.28) - 0.28) < 1e-9
# status(): a gone Pod is forgotten; an idle one is terminated
fake4 = FakeRunPod()
fake4.pods["p9"] = {"id": "p9", "name": "tune-o-clip", "status": "RUNNING"}
api4 = rp.RunPod("KEY123", opener=fake4)
rp.save_state({"pod_id": "p9", "ready": True, "created": 0, "last_activity": 0, "busy": False, "cost_per_hr": 0.28}, state_path)
with quiet() as out:
    assert rp.status(api4, state_path, clock=lambda: 20 * 60) == {}
assert ("DELETE", "/pods/p9", None) in fake4.calls and not state_path.exists() and "idle 20 min" in out.getvalue()
rp.save_state({"pod_id": "ghost", "ready": True, "created": 0}, state_path)
with quiet() as out:
    assert rp.status(api4, state_path) == {} and "is gone" in out.getvalue() and not state_path.exists()
# the live meter, and the spend cap from status()
fake4.pods["p10"] = {"id": "p10", "name": "tune-o-clip", "status": "RUNNING"}
rp.save_state({"pod_id": "p10", "ready": True, "created": 0, "last_activity": 3500, "busy": False, "cost_per_hr": 0.28}, state_path)
with quiet() as out:
    assert rp.status(api4, state_path, clock=lambda: 3600)["pod_id"] == "p10"
ev = json.loads(out.getvalue().strip().splitlines()[-1])
assert ev["session_usd"] == 0.28 and ev["max_usd"] == 1.5 and ev["stage"] == "ready", ev
with quiet() as out:
    rp.save_state(dict(rp.load_state(state_path), busy=True), state_path)
    assert rp.status(api4, state_path, clock=lambda: 6 * 3600) == {}
assert "spend cap" in out.getvalue() and "p10" not in fake4.pods and not state_path.exists(), out.getvalue()
# orphan check (no state on this PC, a tune-o-clip Pod still billing): reported for the red banner, not killed
fake4.pods["orph"] = {"id": "orph", "name": "tune-o-clip", "status": "RUNNING", "cost": 0.28}
fake4.pods["other"] = {"id": "other", "name": "someone-else", "status": "RUNNING", "cost": 1.0}
with quiet() as out:
    assert rp.status(api4, state_path) == {"orphans": ["orph"]}
ev = json.loads(out.getvalue().strip().splitlines()[-1])
assert ev["stage"] == "orphan" and ev["msg"] == "WARNING: RunPod Pod Active ($0.28/hr) - Click to Terminate", ev
with quiet() as out:  # the banner's click = off: every tune-o-clip Pod, nothing else
    assert rp.terminate_cloud_pod(api4, state_path) == ["orph"]
assert "orph" not in fake4.pods and "other" in fake4.pods
# the watchdog ON the Pod: terminates itself when idle, keeps going while a run is busy
fake5 = FakeRunPod()
fake5.pods["me"] = {"id": "me"}
t = iter(range(0, 10 ** 6, 60))
act = tmp / "activity"
act.write_text("")
os.utime(act, (0, 0))
busy = tmp / "busy"
busy.write_text("")
naps = []
def nap(s):
    naps.append(s)
    if len(naps) == 20:
        busy.unlink()  # the run ends after 20 min
with quiet():
    why = rp.watchdog(rp.RunPod("KEY123", opener=fake5), "me", clock=lambda: next(t), sleep=nap, busy=str(busy),
                      activity=str(act), rate=0.28, created=0)
assert why.startswith("idle") and len(naps) == 20 and ("DELETE", "/pods/me", None) in fake5.calls, (why, len(naps))

# --- runs: argv mapping, uploads, pulled paths ---------------------------------------------------------------------
root = tmp / "repo"
(root / "backend").mkdir(parents=True)
(root / "backend" / "clipper.py").write_text("")
(root / "workspace").mkdir()
(root / "workspace" / "twitch_channels.txt").write_text("caseoh_\n")
batch = tmp / "elsewhere" / "batch_urls.txt"
batch.parent.mkdir()
batch.write_text("https://youtu.be/x\n")
args = ["-u", str(root / "backend" / "clipper.py"), "https://youtu.be/x", "--whisper-model", "small.en", "--ollama-model",
        "m3b", "--device", "cuda", "--add-file", str(batch), "--poll-twitch-channels", str(root / "workspace" / "twitch_channels.txt"),
        "--clips", "4"]
mapped, uploads = rp.remote_args(args, root)
assert mapped == ["-u", f"{rp.REMOTE}/backend/clipper.py", "https://youtu.be/x", "--add-file",
                  f"{rp.REMOTE}/workspace/_upload/batch_urls.txt", "--poll-twitch-channels",
                  f"{rp.REMOTE}/workspace/twitch_channels.txt", "--clips", "4"], mapped
assert {r for _, r in uploads} == {f"{rp.REMOTE}/backend/clipper.py", f"{rp.REMOTE}/workspace/_upload/batch_urls.txt",
                                    f"{rp.REMOTE}/workspace/twitch_channels.txt"}, uploads
pulled = root / "output" / "vid1"
pulled.mkdir(parents=True)
(pulled / "clips_manifest.json").write_text(json.dumps({"path": f"{rp.REMOTE}/output/vid1/clip_01.mp4"}))
rp.rewrite_paths(root / "output", root)
assert json.loads((pulled / "clips_manifest.json").read_text())["path"] == root.resolve().as_posix() + "/output/vid1/clip_01.mp4"

# pull_results: a tar stream from the Pod lands under output/ / ready_to_post/, nothing outside the repo
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as tf:
    for name, data in (("ready_to_post/v_1_short.mp4", b"mp4"), ("output/v/clips_manifest.json", json.dumps({"m": rp.REMOTE}).encode())):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
class P:
    def __init__(self, *a, **k):
        self.stdout = io.BytesIO(buf.getvalue())
    def wait(self):
        return 0
assert rp.pull_results({"ip": "x", "port": 1}, "/tmp/m", root, popen=P) == 2
assert (root / "ready_to_post" / "v_1_short.mp4").read_bytes() == b"mp4"
assert json.loads((root / "output" / "v" / "clips_manifest.json").read_text())["m"] == root.resolve().as_posix()
class Empty(P):
    def __init__(self, *a, **k):
        self.stdout = io.BytesIO(b"")
assert rp.pull_results({"ip": "x", "port": 1}, "/tmp/m", root, popen=Empty) == 0
# run_remote refuses without a ready Pod
try:
    rp.run_remote(["-u", "x.py"], tmp / "none.json")
    raise AssertionError("should raise")
except rp.RunPodError as e:
    assert "switch it ON" in str(e)
assert rp.ssh_cmd({"ip": "203.0.113.7", "port": 10341}, "true", tmp)[-2:] == ["root@203.0.113.7", "true"]
# the Pod-side run line: valid shell, busy flag + a heartbeat that dies with its shell, python's exit code is the result
job = rp.job_command(["/root/tune-o-clip/backend/clipper.py", "a b"], "/tmp/toc_run_1")
assert job.endswith("python3 -u /root/tune-o-clip/backend/clipper.py 'a b'") and f"touch {rp.BUSY} /tmp/toc_run_1" in job
assert "while kill -0 $$" in job and f"touch {rp.ACTIVITY}; sleep 30" in job and "</dev/null >/dev/null 2>&1 &" in job
if shutil.which("bash"):  # the mini PC / the Pod (this Windows box has none on PATH)
    assert subprocess.run(["bash", "-n"], input=job.encode(), capture_output=True).returncode == 0, job
    # for real with harmless stand-ins: the heartbeat touches the activity file, then stops after the shell
    hb = tmp / "hb"
    hb.mkdir()
    fake = (job.replace(". /root/toc_env.sh", "true").replace(f"cd {rp.REMOTE}", f"cd '{hb.as_posix()}'")
            .replace(rp.BUSY, "busy").replace(rp.ACTIVITY, "act").replace("/tmp/toc_run_1", "mark").replace("sleep 30", "sleep 0.2")
            .replace("python3 -u /root/tune-o-clip/backend/clipper.py 'a b'", "sleep 1; test -f busy && test -f act || exit 3; exit 7"))
    assert subprocess.run(["bash", "-c", fake], capture_output=True, timeout=20).returncode == 7, "python's exit code comes back"
    assert not (hb / "busy").exists() and (hb / "act").exists(), "the exit trap clears busy and stamps activity"
    __import__("time").sleep(0.6)  # a loop pass already asleep (0.2 s) may touch once more as the shell exits
    stamp = (hb / "act").stat().st_mtime
    __import__("time").sleep(1.0)
    assert (hb / "act").stat().st_mtime == stamp, "the heartbeat stopped with its shell"

# --- .env key: other lines (and a Notepad BOM) survive -------------------------------------------------------------
env = tmp / ".env"
env.write_bytes(b"\xef\xbb\xbfTWITCH_CLIENT_ID=abc\nRUNPOD_API_KEY=old\nX=1\n")
rp.set_env_key("RUNPOD_API_KEY", "new_key", env)
raw = env.read_bytes()
assert raw.startswith(b"\xef\xbb\xbf") and raw.count(b"RUNPOD_API_KEY=") == 1 and b"RUNPOD_API_KEY=new_key" in raw
assert b"TWITCH_CLIENT_ID=abc" in raw and b"X=1" in raw

# --- profile switch: the Pod (TOC_CLOUD=1) vs this PC ---------------------------------------------------------------
probe = ("import sys; sys.path[:0] = [r'%s', r'%s']; import scorer, clipper, pipeline, renderer, json; "
         "print(json.dumps([scorer.CLOUD, scorer.DEFAULT_MODEL, scorer.DEFAULT_WHISPER, scorer.WHISPER_GPU_COMPUTE, "
         "scorer.KEEP_WARM, scorer.LOW_SPEC, pipeline.DEFAULT_MODEL, renderer.detect_encoder('h264_nvenc')[3]]))"
         % (HERE, HERE.parent / "editor"))
def profile(cloud):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TOC_")}
    if cloud:
        env["TOC_CLOUD"] = "1"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, cwd=tmp)
    assert out.returncode == 0, out.stderr[-500:]
    return json.loads(out.stdout.strip().splitlines()[-1])
cloud = profile(True)
assert cloud[:6] == [True, "qwen2.5:14b-instruct", "large-v3-turbo", "float16", "10m", False], cloud
assert cloud[6] == "qwen2.5:14b-instruct" and cloud[7][cloud[7].index("-cq") + 1] == "19", cloud
local = profile(False)
assert local[:4] == [False, "richardyoung/qwen2.5-3b-instruct-abliterated", "small.en", "auto"], local
assert local[7][local[7].index("-cq") + 1] == "22", local

# set-env: only the drawer's non-secret keys
env2 = tmp / ".env2"
env2.write_text("X=1\n")
orig_set = rp.set_env_key
rp.set_env_key = lambda k, v, path=env2: orig_set(k, v, path)
with quiet() as out:
    assert rp.main(["set-env", "RUNPOD_GPU_ID", "NVIDIA GeForce RTX 3090"]) == 0
    assert rp.main(["set-env", "RUNPOD_API_KEY", "nope"]) == 1  # secrets only through stdin (set-key)
    assert rp.main(["set-env", "RUNPOD_IMAGE", "ghcr.io/tune-0-matic/tune-o-clip-pod:v1"]) == 0  # the build script's last step
    assert rp.main(["set-env", "RUNPOD_IMAGE", "x\nRUNPOD_API_KEY=evil"]) == 1, "one line only: no smuggled .env keys"
    assert rp.main(["set-env", "RUNPOD_IMAGE", "  "]) == 1
    # the drawer saves both choices in ONE process: a GPU pick and a cloud-type change can't overwrite each other
    assert rp.main(["set-env", "RUNPOD_GPU_ID", "NVIDIA A40", "RUNPOD_CLOUD_TYPE", "SECURE"]) == 0
    assert rp.main(["set-env", "RUNPOD_GPU_ID", "NVIDIA L4", "RUNPOD_API_KEY", "nope"]) == 1, "validated before any write"
    assert rp.main(["set-env", "RUNPOD_GPU_ID"]) == 1 and rp.main(["set-env"]) == 1
rp.set_env_key = orig_set
saved = env2.read_text()
assert "RUNPOD_GPU_ID=NVIDIA A40" in saved and "RUNPOD_CLOUD_TYPE=SECURE" in saved and "NVIDIA L4" not in saved, saved
assert saved.count("RUNPOD_GPU_ID=") == 1 and "nope" not in saved and "X=1" in saved, saved
assert "RUNPOD_IMAGE=ghcr.io/tune-0-matic/tune-o-clip-pod:v1" in saved and "evil" not in saved
assert not (tmp / ".env2.tmp").exists(), "written atomically: the temp file is swapped in, never left behind"
# a reader racing the write (an ON right after a click) sees the old or the new .env, never a truncated one
real_replace = os.replace
seen = []
os.replace = lambda a, b: (seen.append((Path(b).read_text(), Path(a).read_text())), real_replace(a, b))[1]
try:
    rp.set_env_key("RUNPOD_GPU_ID", "NVIDIA RTX 4090", env2)
finally:
    os.replace = real_replace
(before, after), = seen
assert "RUNPOD_GPU_ID=NVIDIA A40" in before and "RUNPOD_GPU_ID=NVIDIA RTX 4090" in after and "X=1" in after, seen
# stats: the Pod's sample (SSH faked) carries the GPU type and price for the sidebar
rp.save_state({"pod_id": "s1", "ready": True, "ip": "203.0.113.7", "port": 1, "gpu": "NVIDIA RTX 4000 Ada Generation", "cost_per_hr": 0.28}, state_path)
class R:
    stdout = json.dumps({"cpu": 37.5, "vcpu": 9.0, "ram_used": 12.3, "ram_total": 50.0, "disk_free": 18.4,
                         "gpu": {"name": "NVIDIA RTX 4000 Ada Generation", "util": 61.0, "used": 9100.0, "total": 20475.0, "temp": 58.0}}) + "\n"
sent = {}
def fake_run(cmd, **kw):
    sent.update(cmd=cmd, input=kw.get("input"))
    return R()
s = rp.pod_stats(state_path, run=fake_run)
assert s["gpu"]["total"] == 20475.0 and s["cost_per_hr"] == 0.28 and s["gpu_type"].startswith("NVIDIA RTX 4000") and s["vcpu"] == 9.0
assert sent["cmd"][-1] == "python3 -" and b"memory.max" in sent["input"] and b"nvidia-smi" in sent["input"]
state_path.unlink()
assert rp.pod_stats(state_path, run=fake_run) == {}, "no Pod: no SSH at all"

# --- 429 rate limits (live 2026-11-15): a 429 mid-start read as "Pod gone", and the cleanup terminate 429'd too ------
class Limited(FakeRunPod):
    """Answers the first n requests of each (method, path-kind) with 429, then behaves."""
    def __init__(self, n=2, **kw):
        super().__init__(**kw)
        self.left = {}
        self.n = n
    def __call__(self, req, timeout=None):
        kind = (req.get_method(), req.full_url.replace(rp.API, "").split("/")[1] if "/pods" in req.full_url else req.full_url)
        if self.left.setdefault(kind, self.n) > 0:
            self.left[kind] -= 1
            self.calls.append((req.get_method(), req.full_url.replace(rp.API, ""), "429"))
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many", {}, io.BytesIO(b'{"error":"rate limit exceeded for the minute window"}'))
        return super().__call__(req, timeout)
naps = []
lim = Limited(n=3)
lim.pods["p429"] = {"id": "p429", "name": "tune-o-clip", "status": "RUNNING"}
api_l = rp.RunPod("KEY123", opener=lim, sleep=naps.append)
with quiet() as out:
    api_l.terminate("p429")  # 3x 429, then 204
assert "p429" not in lim.pods and naps == [5, 10, 20] and out.getvalue().count('"retry"') == 3, (naps, lim.calls[-4:])
# the start poll rides out 429s instead of declaring the Pod gone
lim2 = Limited(n=2)
lim2.pods["px"] = {"id": "px", "name": "tune-o-clip", "status": "RUNNING", "ssh": None}
with quiet():
    st429 = rp.wait_ready(rp.RunPod("KEY123", opener=lim2, sleep=lambda s: None), {"pod_id": "px"}, sleep=lambda s: None)
assert st429["ip"] == "203.0.113.7", st429
# a create is never retried (a 5xx may come after it was made); v2's 400 "no instances" moves on to the next placement
class NoInst(FakeRunPod):
    def __call__(self, req, timeout=None):
        if req.get_method() == "POST" and req.full_url.endswith("/pods"):
            self.calls.append(("POST", "/pods", json.loads(req.data)))
            raise urllib.error.HTTPError(req.full_url, 400, "x", {}, io.BytesIO(b'{"detail":"There are no instances currently available","status":400}'))
        return super().__call__(req, timeout)
ni = NoInst()
with quiet() as out:
    try:
        rp.start_cloud_pod(rp.RunPod("KEY123", opener=ni, sleep=lambda s: (_ for _ in ()).throw(AssertionError("a create was retried"))),
                           "KEY123", None, tmp / "ni.json", ssh_ready=None, do_upload=None, do_bootstrap=None, sleep=lambda s: None)
        raise AssertionError("should raise")
    except rp.NoCapacity as e:
        assert "no instances" in str(e) and "refused every option" in str(e), str(e)
posts = [c for c in ni.calls if c[:2] == ("POST", "/pods")]
assert len(posts) == len({(c[2]["gpu"]["id"], c[2]["cloud"]) for c in posts}) > 1, "one POST per placement, then the next"
# persistent errors raise (never "gone"); a real 404 is gone
always = Limited(n=99)
try:
    with quiet():
        rp.RunPod("KEY123", opener=always, sleep=lambda s: None).get("px")
    raise AssertionError("should raise")
except rp.RunPodError as e:
    assert "HTTP 429" in str(e)
assert rp.RunPod("KEY123", opener=FakeRunPod()).get("nope") is None
# a cleanup terminate that keeps failing: loud error, state KEPT so OFF / status can still end the Pod
class StuckDelete(FakeRunPod):
    def __call__(self, req, timeout=None):
        if req.get_method() == "DELETE":
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many", {}, io.BytesIO(b"rate limited"))
        return super().__call__(req, timeout)
sd = StuckDelete()
sd_state = tmp / "sd_state.json"
def boom(s):
    raise rp.RunPodError("bootstrap died")
with quiet() as out:
    try:
        rp.start_cloud_pod(rp.RunPod("KEY123", opener=sd, sleep=lambda s: None), "KEY123", None, sd_state,
                           ssh_ready=lambda s: None, do_upload=lambda s: None, do_bootstrap=boom, sleep=lambda s: None)
        raise AssertionError("should raise")
    except rp.RunPodError:
        pass
assert "COULD NOT TERMINATE" in out.getvalue() and json.loads(sd_state.read_text())["pod_id"] == "pod1", "state kept for OFF"

# --- LF only to the Pod (regression: CRLF bootstrap -> "set: - : invalid option", "cd: $'/root/tune-o-clip\r'") ----
st = {"ip": "203.0.113.7", "port": 1, "cost_per_hr": 0.59, "created": 1000.0}
payload = rp.bootstrap_script(st)
assert b"\r" not in payload and payload.startswith(b"set -e\ncd /root/tune-o-clip\n"), payload[:60]
assert b"trap 'say error" in payload and b"$BASH_COMMAND" in payload and b">/root/pull.log" in payload, "a failing step names itself"


def branch(needle):
    """Which bootstrap path runs the line holding needle: "baked" (docker/Dockerfile.runpod image), "generic" or "both"."""
    state = "both"
    for line in payload.decode().splitlines():
        s = line.strip()
        if s.startswith('if [ "$BAKED" = 1 ]'):
            state = "baked"
        elif s.startswith('if [ "$BAKED" = 0 ]'):
            state = "generic"
        elif s == "else" and state == "baked":
            state = "generic"
        elif s == "fi":
            state = "both"
        if needle in line and not s.startswith("#"):
            return state


assert b'[ -f /opt/toc_baked ] && command -v ffmpeg >/dev/null && command -v ollama >/dev/null; then BAKED=1' in payload
assert [branch(n) for n in ("apt-get update", "ollama.com/install.sh", "ollama pull", "WhisperModel(")] == ["generic"] * 4, \
    "the pre-baked image skips apt, the Ollama installer, the 9 GB pull and the Whisper download"
assert [branch(n) for n in ("ollama serve", "> /root/toc_env.sh", "watchdog", 'say ready')] == ["both"] * 4
assert branch("export OLLAMA_MODELS=/opt/ollama/models HF_HOME=/opt/hf_cache  #") == "baked"
assert b"echo 'export OLLAMA_MODELS=/opt/ollama/models HF_HOME=/opt/hf_cache' >> /root/toc_env.sh" in payload, "runs find the baked models"
if shutil.which("bash"):
    assert subprocess.run(["bash", "-n"], input=payload, capture_output=True).returncode == 0, "bootstrap is valid bash"
# the image: models where the bootstrap looks, the marker it checks, no repo files (keys / cookies) baked in
dockerfile = (Path(rp.__file__).resolve().parent.parent / "docker" / "Dockerfile.runpod").read_text(encoding="utf-8")
assert f"FROM {rp.IMAGE}" in dockerfile and "OLLAMA_MODELS=/opt/ollama/models HF_HOME=/opt/hf_cache" in dockerfile
assert f"ARG MODEL={rp.CLOUD_MODEL}" in dockerfile and f"ARG WHISPER={rp.CLOUD_WHISPER}" in dockerfile and "/opt/toc_baked" in dockerfile
assert [l.split()[1] for l in dockerfile.splitlines() if l.startswith(("COPY", "ADD"))] == ["requirements.txt"], "only requirements.txt"
assert b"TOC_RUNPOD_RATE=0.59 TOC_RUNPOD_CREATED=1000" in payload and b"runpod_manager.py watchdog" in payload
assert b"/proc/1/environ" in payload and b"TOC_RUNPOD_KEY" in payload, "the watchdog gets the Pod's env (SSH sessions don't)"
assert b"|| say warn" in payload and b"whisper_warmup.log" in payload, "a Whisper warm-up failure must not end the bootstrap"
assert b"/root/toc_encoder" in payload and b"/root/toc_encoder" in rp.lf(rp.POD_STATS), "the Pod reports its real encoder"
import ast as _ast
_ast.parse(rp.POD_STATS)  # the script that runs on the Pod is valid Python (cgroup v1 + v2 paths)
assert b"cpuacct.usage" in rp.lf(rp.POD_STATS) and b"memory.limit_in_bytes" in rp.lf(rp.POD_STATS)
assert payload.index(b"/proc/1/environ") < payload.index(b"runpod_manager.py watchdog"), "env written before the watchdog"
class CapturePopen:
    cmds = []
    def __init__(self, cmd, stdin=None, stdout=None, stderr=None, **kw):
        assert "text" not in kw and "encoding" not in kw, "binary pipes only: Windows text mode adds \\r"
        assert stdin == subprocess.DEVNULL, "the running script gets no stdin to read"
        CapturePopen.cmds.append(cmd)
        self.stdout = iter([b'{"event": "runpod", "stage": "ready", "msg": "Ready"}\n'])
    def wait(self):
        return 0
copied = []
def capture_copy(cmd, input=None, **kw):
    assert isinstance(input, bytes) and "text" not in kw, "the tar goes up as bytes"
    copied.append((cmd, input))
    return subprocess.CompletedProcess(cmd, 0, b"", b"")
with quiet():
    rp.bootstrap(st, popen=CapturePopen, run=capture_copy)
(copy_cmd, tar_bytes), = copied
assert copy_cmd[-1] == "tar xf - -C /tmp", copy_cmd
with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
    (member,) = tf.getmembers()
    script = tf.extractfile(member).read()
assert member.name == "toc_bootstrap.sh" and member.mode == 0o755 and script == payload and b"\r" not in script
(run_cmd,) = CapturePopen.cmds
assert run_cmd[-1] == "bash /tmp/toc_bootstrap.sh </dev/null", run_cmd
assert not any("bash -s" in str(c) for c in [copy_cmd, run_cmd]), "never pipe the script through bash -s"
failed = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, b"", b"No space left on device")
try:
    rp.bootstrap(st, popen=CapturePopen, run=failed)
    raise AssertionError("a failed copy must stop before running anything")
except rp.RunPodError as e:
    assert "No space left" in str(e) and len(CapturePopen.cmds) == 1, e
# the upload: a Windows working copy (CRLF files, a Notepad BOM .env) arrives LF-only; scripts executable
win = tmp / "winrepo"
(win / "backend").mkdir(parents=True)
(win / "workspace").mkdir()
(win / "backend" / "runpod_manager.py").write_bytes(b"import os\r\nprint('watchdog')\r\n")  # what runs the watchdog
(win / "setup_linux.sh").write_bytes(b"#!/usr/bin/env bash\r\nset -e\r\n")
(win / "assets.bin").write_bytes(b"\x00\x01\r\n\x02")  # binary: untouched
(win / ".env").write_bytes(b"\xef\xbb\xbfTWITCH_CLIENT_ID=abc\r\nX=1\r\nYOUTUBE_CLIENT_SECRET=yt\r\nRUNPOD_API_KEY=rpa_x\r\n"
                           b"TWITCH_CLIENT_SECRET=def\r\n")
(win / "cookies.txt").write_bytes(b"# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tsecret\n")
(win / "workspace" / "settings.json").write_bytes(b'\xef\xbb\xbf{\r\n "whisper_model": "small.en",\r\n "caption_theme": "x"\r\n}')
subprocess.run(["git", "init", "-q"], cwd=win, check=True)
subprocess.run(["git", "-c", "core.autocrlf=false", "add", "setup_linux.sh", "assets.bin"], cwd=win, check=True)  # runpod_manager.py untracked: must go up anyway
got_tar = {}
def capture_run(cmd, stdin=None, **kw):
    got_tar["data"] = stdin.read()
    return subprocess.CompletedProcess(cmd, 0, b"", b"")
rp.upload(st, root=win, run=capture_run)
members = {}
with tarfile.open(fileobj=io.BytesIO(got_tar["data"])) as tf:
    for m in tf.getmembers():
        members[m.name] = (tf.extractfile(m).read(), m.mode)
assert set(members) >= {"backend/runpod_manager.py", "setup_linux.sh", ".env", "workspace/settings.json", "assets.bin"}, set(members)
for name, (data, mode) in members.items():
    if name != "assets.bin":
        assert b"\r" not in data and not data.startswith(b"\xef\xbb\xbf"), (name, data[:40])
assert members["backend/runpod_manager.py"][0] == b"import os\nprint('watchdog')\n"
assert members[".env"][0] == b"TWITCH_CLIENT_ID=abc\nTWITCH_CLIENT_SECRET=def\n" and members["setup_linux.sh"][1] == 0o755
assert members[".env"][1] == 0o600 and "cookies.txt" not in members, "the Pod never gets the YouTube login / other secrets"
assert not any(b"rpa_x" in d or b"SID\tsecret" in d or b"=yt" in d for d, _ in members.values()), "a secret leaked"
assert members["assets.bin"][0] == b"\x00\x01\r\n\x02", "binary files are not rewritten"
assert "whisper_model" not in json.loads(members["workspace/settings.json"][0]), "the Pod uses its cloud models"
assert not (win / "workspace" / "runpod_upload.tar").exists()
# the stats script and uploaded run arguments: bytes, LF
sent_stats = {}
def stats_run(cmd, input=None, **kw):
    sent_stats["input"] = input
    assert "text" not in kw
    return subprocess.CompletedProcess(cmd, 0, b'{"cpu": 1, "gpu": {}}\n', b"")
rp.save_state(dict(st, ready=True), state_path)
assert rp.pod_stats(state_path, run=stats_run)["cpu"] == 1 and isinstance(sent_stats["input"], bytes) and b"\r" not in sent_stats["input"]
batch_crlf = tmp / "batch_crlf.txt"
batch_crlf.write_bytes(b"https://youtu.be/a\r\nhttps://youtu.be/b\r\n")
sent_args = []
class RunP:
    def __init__(self, *a, **k):
        self.stdout = io.BytesIO(b"")  # the run's stream, then the (empty) results tar
    def wait(self):
        return 0
def args_run(cmd, input=None, **kw):
    sent_args.append(input)
    return subprocess.CompletedProcess(cmd, 0, b"", b"")
with quiet():
    rp.run_remote(["-u", str(root / "backend" / "clipper.py"), "--add-file", str(batch_crlf)], state_path, root, popen=RunP, run=args_run)
assert b"https://youtu.be/a\nhttps://youtu.be/b\n" in sent_args and all(b"\r" not in s for s in sent_args if s), sent_args
state_path.unlink()

# main(): no key -> status stays quiet (most sessions never use RunPod); no network was touched anywhere
with quiet() as out:
    assert rp.main(["status"]) == 0
assert "no RunPod key" in out.getvalue()
with quiet() as out:
    assert rp.main(["on"]) == 1 and "RUNPOD_API_KEY is not set" in out.getvalue()
assert blocked == [], blocked
assert os.environ["RUNPOD_API_KEY"] == "", "the real key was never loaded"

print("runpod ok")
