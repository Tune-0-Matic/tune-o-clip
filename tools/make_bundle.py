"""Build ../tune-o-clip-bundle.zip for Google Drive: git archive of HEAD (LF line endings, exec bits) + .env, env.txt
(visible copy) and cookies.txt + a .git that already points at the GitHub remote, so `git pull` works after unzip.

  .venv/Scripts/python tools/make_bundle.py        (commit first: the bundle is HEAD, not the working tree)

The zip holds your keys and YouTube login cookies: keep it private. Prints names only, never file contents."""
import io
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
out = Path(sys.argv[2]) if len(sys.argv) > 2 else repo.parent / "tune-o-clip-bundle.zip"
P = "tune-o-clip/"
url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
tar_bytes = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=repo, capture_output=True, check=True).stdout


def add(z, name, data, mode=0o644):
    info = zipfile.ZipInfo(P + name, time.localtime()[:6])
    info.external_attr = (0o100000 | mode) << 16  # regular file + permissions: unzip on Linux restores them
    info.create_system = 3  # Unix: otherwise (written on Windows = MS-DOS) unzip ignores the permission bits
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, data)


tmp = out.with_suffix(".tmp")
with zipfile.ZipFile(tmp, "w") as z:
    n_code = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as t:
        for m in t.getmembers():
            if m.isfile():
                add(z, m.name, t.extractfile(m).read(), m.mode & 0o777)
                n_code += 1
    secrets = []
    env = repo / ".env"
    if env.exists():
        add(z, ".env", env.read_bytes(), 0o600)
        add(z, "env.txt", env.read_bytes(), 0o600)  # visible copy; setup_linux.sh renames it if .env is missing
        secrets += [".env", "env.txt"]
    if (repo / "cookies.txt").exists():
        add(z, "cookies.txt", (repo / "cookies.txt").read_bytes(), 0o600)
        secrets.append("cookies.txt")
    # .git: objects + refs as they are, a clean Linux config (no Windows core.* settings, no local identity)
    git = repo / ".git"
    n_git = 0
    for f in sorted(git.rglob("*")):
        rel = f.relative_to(repo).as_posix()
        if f.is_file() and rel not in (".git/config", ".git/index") and not rel.startswith(".git/hooks/"):
            add(z, rel, f.read_bytes(), 0o644)
            n_git += 1
    # a stat-free index of HEAD (this PC's index holds Windows sizes / CRLF stats): git just re-hashes on first use
    idx = repo / ".git" / "bundle_index.tmp"
    subprocess.run(["git", "read-tree", "HEAD"], cwd=repo, check=True, env={**__import__("os").environ, "GIT_INDEX_FILE": str(idx)})
    add(z, ".git/index", idx.read_bytes(), 0o644)
    idx.unlink()
    add(z, ".git/config", "[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n\tbare = false\n\tlogallrefupdates = true\n"
        f'[remote "origin"]\n\turl = {url}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n'
        '[branch "main"]\n\tremote = origin\n\tmerge = refs/heads/main\n')
tmp.replace(out)
print(f"{out}: {out.stat().st_size / 1e6:.2f} MB, {n_code} code files from HEAD, {n_git} .git files, secrets: {', '.join(secrets) or 'none'}")
