#!/bin/sh
# install_from_local.sh —— 从 portal pkgs（已下载的 357 包）离线建环境。
# 步骤：portal pkgs → 本地 channel（按 URL 子目录分 linux-64/noarch，
# 从包内 info/index.json 生成最小 repodata.json）→ micromamba create
# -c file:// 本地 channel（秒级求解、零网络）→ rebuild_renv.sh 收尾。
# 幂等：被杀后重跑即可（本地随重启丢失，portal pkgs 持久）。

set -eu
PORTAL=/mnt/agents/output/r_env
SIM=/mnt/agents/output/code/simulation
# 工作根：默认 $HOME；容器周期性重置会清 $HOME 但保留 /tmp（实测），
# 用 R_WORK_DIR=/tmp/rwork 让环境跨重置存活（调用方负责符号链接）
WORK="${R_WORK_DIR:-$HOME}"
MM="$WORK/bin/micromamba"
export MAMBA_ROOT_PREFIX="$WORK/micromamba"

mkdir -p "$WORK/bin"
[ -x "$MM" ] || { cp "$PORTAL/bin/micromamba" "$MM"; chmod +x "$MM"; }

echo "[install] 同步 portal pkgs → 本地 channel ..."
mkdir -p "$WORK/chan/linux-64" "$WORK/chan/noarch"
python3 - <<'PYEOF'
import hashlib
import json, os, shutil
st = json.load(open("/mnt/agents/output/r_env/dl_state.json"))
home = os.environ.get("R_WORK_DIR", os.path.expanduser("~"))
rep = {"linux-64": {}, "noarch": {}}
for name, v in st.items():
    fn = v["fn"]
    subdir = "noarch" if "/noarch/" in v["url"] else "linux-64"
    src = os.path.join("/mnt/agents/output/r_env/pkgs", fn)
    dst = os.path.join(home, "chan", subdir, fn)
    if not (os.path.exists(dst) and os.path.getsize(dst) == v["size"]):
        shutil.copyfile(src, dst)
    # 掐死 post-link 慢下载：genomeinfodbdata/data-packages 的 post-link
    # 会从 bioconductor.org 直连拉 12MB（~20KB/s，必超重启窗口）。
    # 改写包内 post-link 为 no-op，数据包以 GIDD 桩包替代（见 3a）。
    if fn.endswith(".tar.bz2") and (
            "genomeinfodbdata" in fn or "bioconductor-data-packages" in fn):
        import tarfile, io as _io
        tmp_out = dst + ".new"
        with tarfile.open(dst, "r:bz2") as tin, \
                tarfile.open(tmp_out, "w:bz2") as tout:
            for m in tin:
                if m.name.endswith("post-link.sh"):
                    data = b"#!/bin/sh\nexit 0\n"
                    m.size = len(data)
                    tout.addfile(m, _io.BytesIO(data))
                else:
                    tout.addfile(m, tin.extractfile(m))
        os.replace(tmp_out, dst)
        v = dict(v); v["size"] = os.path.getsize(dst)
    name0, ver, build = fn.rsplit("-", 2)
    build = build.replace(".tar.bz2", "").replace(".conda", "")
    h_md5 = hashlib.md5()
    h_sha = hashlib.sha256()
    with open(dst, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h_md5.update(chunk)
            h_sha.update(chunk)
    # 全量钉版安装（357 个 name=version=build 显式指定），依赖闭包
    # 已由锁文件保证，repodata 无需真实 depends（免解析 .conda 元数据）；
    # md5/sha256 必须提供，否则 micromamba 丢弃 .tar.bz2 记录
    rep[subdir][fn] = {
        "name": name0, "version": ver,
        "build": build, "build_number": 0,
        "depends": [], "size": v["size"], "subdir": subdir,
        "md5": h_md5.hexdigest(), "sha256": h_sha.hexdigest(),
    }
for sd, pkgs in rep.items():
    rd = {"info": {"subdir": sd}, "packages": {}, "packages.conda": {}}
    for fn, meta in pkgs.items():
        key = "packages.conda" if fn.endswith(".conda") else "packages"
        rd[key][fn] = meta
    path = os.path.join(home, "chan", sd, "repodata.json")
    with open(path, "w") as f:
        json.dump(rd, f)
    print(f"[install] repodata {sd}: {len(pkgs)} 包")
# 全量钉版 spec 文件
with open(os.path.join(home, "full_specs.txt"), "w") as f:
    for line in open("/mnt/agents/output/code/simulation/renv_lock.yml"):
        line = line.strip()
        if line.startswith("- ") and "=" in line:
            f.write(line[2:] + "\n")
print("[install] full_specs.txt 就绪")
PYEOF

echo "[install] micromamba create（本地 channel + 全量钉版，离线）..."
rm -rf "$MAMBA_ROOT_PREFIX/envs/renv"
MAMBA_ADD_PIP_AS_PYTHON_DEPENDENCY=false \
"$MM" create -y -n renv -c "file://$WORK/chan" --override-channels \
  --file "$WORK/full_specs.txt"

echo "[install] create 完成，收尾（GIDD 桩包 / microbiome / 版本核对）..."
P="$MAMBA_ROOT_PREFIX/envs/renv"
# 3a. GIDD 桩包：替代 GenomeInfoDbData 数据包（真包需 12MB 慢速下载，
#     且 DESeq2/ANCOM-BC2 运行时不使用其基因组数据）。桩包含
#     DESCRIPTION/NAMESPACE/zzz.R，使 library() 与 packageVersion() 正常。
if [ ! -d "$P/lib/R/library/GenomeInfoDbData" ]; then
  echo "[install] 安装 GIDD 桩包 ..."
  STUB="$WORK/gidd_stub/GenomeInfoDbData"
  mkdir -p "$STUB/R"
  cat > "$STUB/DESCRIPTION" <<'EOF'
Package: GenomeInfoDbData
Version: 1.2.13
Title: Stub for GenomeInfoDbData (data package, contents not required)
Description: Minimal stub replacing the Bioconductor annotation data
    package GenomeInfoDbData. Installed to satisfy the package dependency
    of GenomeInfoDb without the 12MB data download; species/assembly
    lookup tables are not used by the ZOLA baseline methods.
License: Artistic-2.0
Encoding: UTF-8
EOF
  echo '# stub: no R objects exported' > "$STUB/NAMESPACE"
  cat > "$STUB/R/zzz.R" <<'EOF'
.onLoad <- function(libname, pkgname) {
    packageStartupMessage("GenomeInfoDbData stub (no data tables)")
}
EOF
  # 修复（19fb27ff 备忘 bug）：install.packages(repos=NULL) 不能装源码
  # 目录（仅接受 tarball），且 Sys.getenv("R_WORK_DIR") 未导出时拼出
  # 错路径，|| true 静默吞错。改用 R CMD INSTALL（可直接装目录）+ 硬校验。
  "$P/bin/R" CMD INSTALL "$STUB"
  if [ ! -d "$P/lib/R/library/GenomeInfoDbData" ]; then
    echo "[install] !! GIDD 桩包安装失败" >&2
    exit 1
  fi
fi
# 3b. microbiome（ANCOMBC data_sanity_check 依赖；本地 tarball 优先，
#     备份在 portal r_env/srcpkg/，避免 bioconductor.org 慢速直连）
MB_SRC="$PORTAL/srcpkg/microbiome_1.28.0.tar.gz"
if ! "$MM" run -n renv Rscript -e 'library(microbiome)' >/dev/null 2>&1; then
  echo "[install] 安装 microbiome（本地 tarball）..."
  [ -f "$MB_SRC" ] || curl -sL --retry 3 -o "$MB_SRC" \
    "https://bioconductor.org/packages/3.20/bioc/src/contrib/microbiome_1.28.0.tar.gz"
  "$MM" run -n renv Rscript -e \
    'install.packages("'"$MB_SRC"'", repos = NULL, type = "source")' || true
fi
# 4. 版本核对
"$MM" run -n renv Rscript -e '
cat("R:", R.version.string, "\n")
for (p in c("ANCOMBC", "corncob", "MicrobiomeStat", "DESeq2", "phyloseq",
            "microbiome", "GenomeInfoDbData"))
  cat(sprintf("%-15s %s\n", p, as.character(packageVersion(p))))
'
echo "INSTALL_DONE"
