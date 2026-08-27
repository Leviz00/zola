#!/usr/bin/env python3
"""dl_packages.py —— 免求解、可断点续传的 renv 全量包下载器。

从 renv_lock.yml 解析 name=version=build（357 包），按候选
(channel, subdir, ext) 组合 HEAD 探测真实 URL（结果缓存到 portal
state.json），然后多线程 Range 续传下载到 portal pkgs 目录。
幂等：反复运行直至打印 ALL_DONE。被杀后已下载字节保留（portal 纯文件），
重启后从 state.json + 文件大小继续。

用法：python3 dl_packages.py
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import urllib.request
import urllib.error

PORTAL = "/mnt/agents/output/r_env"
LOCK = "/mnt/agents/output/code/simulation/renv_lock.yml"
CACHE = os.path.join(PORTAL, "pkgs")          # 最终 tarball 存放（portal 纯文件）
TMP = os.path.expanduser("~/dl_part")         # 下载中（本地 ext4，快；重启丢
                                              # 在飞文件，单包均 ~2MB 损失可忽略）
STATE = os.path.join(PORTAL, "dl_state.json")
CHANNELS = ["bioconda", "conda-forge"]
SUBDIRS = ["noarch", "linux-64"]
EXTS = [".conda", ".tar.bz2"]
BASE = "https://conda.anaconda.org/{ch}/{sd}/{fn}"
UA = {"User-Agent": "micromamba/2.8.1"}
THREADS = 10


def parse_lock():
    pkgs = []
    for line in open(LOCK):
        line = line.strip()
        if line.startswith("- ") and "=" in line:
            spec = line[2:].strip()
            name, ver, build = spec.split("=")
            pkgs.append((name, ver, build))
    return pkgs


def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE))
        except Exception:
            return {}
    return {}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE)


def head(url, timeout=20):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return int(r.headers.get("Content-Length", -1))
    except Exception:
        return -1


def resolve(pkg):
    """探测真实 URL，返回 (fn, url, size) 或 None。"""
    name, ver, build = pkg
    stem = f"{name}-{ver}-{build}"
    for ch in CHANNELS:
        for sd in SUBDIRS:
            for ext in EXTS:
                fn = stem + ext
                url = BASE.format(ch=ch, sd=sd, fn=fn)
                size = head(url)
                if size > 0:
                    return fn, url, size
    return None


def _finish(dst_part, dst):
    """跨文件系统完成落盘（本地 .part → portal pkgs）。"""
    with open(dst_part, "rb") as fi, open(dst, "wb") as fo:
        while True:
            chunk = fi.read(1 << 20)
            if not chunk:
                break
            fo.write(chunk)
        fo.flush()
        os.fsync(fo.fileno())
    os.remove(dst_part)


def valid_archive(path):
    """归档完整性校验：bz2 全量解码 / conda zip CRC。

    系统 python 的 tarfile 缺 bz2 模块（容器裁剪），对 .tar.bz2 一律
    误报 False 导致反复重下；改用外部 tar 命令全量解码判定 rc。
    """
    import subprocess
    import zipfile
    try:
        if path.endswith(".tar.bz2"):
            with open(os.devnull, "wb") as dn:
                rc = subprocess.run(["tar", "-tjf", path],
                                    stdout=dn, stderr=dn).returncode
            return rc == 0
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad is not None:
                return False
        return True
    except Exception:
        return False


def download(url, dst_part, dst, size, max_retries=8):
    """Range 续传下载到本地 .part，校验归档后拷入 portal。"""
    for attempt in range(max_retries):
        have = os.path.getsize(dst_part) if os.path.exists(dst_part) else 0
        if have == size and valid_archive(dst_part):
            _finish(dst_part, dst)
            return True
        if have >= size:  # 大小已到但校验不过 / 异常 → 重下
            os.remove(dst_part)
            have = 0
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    # 服务器忽略 Range（返回全量 200）→ 从头写，避免拼接损坏
                    mode = "wb"
                else:
                    mode = "ab"
                with open(dst_part, mode) as f:
                    while True:
                        chunk = r.read(1 << 18)
                        if not chunk:
                            break
                        f.write(chunk)
            # 下完一轮检查：大小 + 归档完整性
            if os.path.getsize(dst_part) == size and valid_archive(dst_part):
                _finish(dst_part, dst)
                return True
        except Exception:
            time.sleep(min(2 ** attempt, 30))
    return False


def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    pkgs = parse_lock()
    st = load_state()
    t0 = time.time()
    done = [k for k, v in st.items() if v.get("done")]
    print(f"[dl] 锁定 {len(pkgs)} 包，已完成 {len(done)}", flush=True)

    # 1. 解析 URL（并发 HEAD；结果入 state）
    def res_one(pkg):
        name = pkg[0]
        if st.get(name, {}).get("done") or st.get(name, {}).get("url"):
            return name, st.get(name)
        r = resolve(pkg)
        if r is None:
            return name, {"error": "NOT_FOUND"}
        fn, url, size = r
        return name, {"fn": fn, "url": url, "size": size, "done": False}

    todo = [p for p in pkgs if not st.get(p[0], {}).get("done")]
    with ThreadPoolExecutor(THREADS) as ex:
        for name, info in ex.map(res_one, todo):
            st[name] = info
    save_state(st)
    errs = [k for k, v in st.items() if v.get("error")]
    if errs:
        print(f"[dl] !! 未找到 URL 的包: {errs}", flush=True)

    # 2. 下载（并发续传）
    def dl_one(name):
        info = st[name]
        if info.get("done") or info.get("error"):
            return name, True
        dst = os.path.join(CACHE, info["fn"])
        part = os.path.join(TMP, info["fn"] + ".part")
        # 已完成但 state 未标（上次 rename 后被杀）
        if os.path.exists(dst) and os.path.getsize(dst) == info["size"]:
            return name, True
        ok = download(info["url"], part, dst, info["size"])
        return name, ok

    names = [k for k, v in st.items() if not v.get("done") and not v.get("error")]
    finished = 0
    with ThreadPoolExecutor(THREADS) as ex:
        for name, ok in ex.map(dl_one, names):
            if ok:
                st[name]["done"] = True
                finished += 1
                if finished % 20 == 0:
                    save_state(st)
                    print(f"[dl] 进度 {sum(1 for v in st.values() if v.get('done'))}"
                          f"/{len(pkgs)} ({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"[dl] !! 下载失败 {name}", flush=True)
    save_state(st)
    n_done = sum(1 for v in st.values() if v.get("done"))
    print(f"[dl] 本轮结束：{n_done}/{len(pkgs)} ({time.time()-t0:.0f}s)",
          flush=True)
    if n_done == len(pkgs) and not errs:
        print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
