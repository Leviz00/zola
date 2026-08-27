#!/bin/sh
# finish_env.sh —— 一链到底：install_from_local.sh → tar → portal（写探测
# + md5 复读校验）。中途被杀则整体重跑（各步幂等）。
export R_WORK_DIR=/tmp/rwork
mkdir -p /tmp/rwork
P=/mnt/agents/output/r_env

echo "$(date '+%F %T') [finish] 安装开始"
sh /mnt/agents/output/code/simulation/install_from_local.sh >> /tmp/install_v3.log 2>&1 || {
  echo "$(date '+%F %T') [finish] !! install 失败，见 /tmp/install_v3.log"; exit 1; }
[ -x /tmp/rwork/micromamba/envs/renv/bin/Rscript ] || {
  echo "$(date '+%F %T') [finish] !! Rscript 缺失"; exit 1; }

echo "$(date '+%F %T') [finish] 安装完成，打包中 ..."
tar -czf /tmp/renv.tar.gz -C /tmp/rwork micromamba/envs/renv || {
  echo "$(date '+%F %T') [finish] !! tar 失败"; exit 1; }

for i in 1 2 3 4 5; do
  echo probe > "$P/.probe_env" 2>/dev/null && \
    [ "$(cat "$P/.probe_env" 2>/dev/null)" = "probe" ] && break
  sleep 10
done
rm -f "$P/.probe_env"
cp /tmp/renv.tar.gz "$P/renv.tar.gz" || {
  echo "$(date '+%F %T') [finish] !! portal 拷贝失败"; exit 1; }

m1=$(md5sum /tmp/renv.tar.gz | cut -d' ' -f1)
for i in 1 2 3 4 5 6; do
  m2=$(md5sum "$P/renv.tar.gz" 2>/dev/null | cut -d' ' -f1)
  [ "$m1" = "$m2" ] && { echo "$(date '+%F %T') ENV_TARBALL_OK md5=$m1"; exit 0; }
  sleep 15
done
echo "$(date '+%F %T') [finish] !! md5 复读不一致 local=$m1 portal=$m2"
exit 1
