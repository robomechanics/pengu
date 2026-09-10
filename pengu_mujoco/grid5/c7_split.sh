#!/usr/bin/env bash
# Automatic c7 three-machine phi split (Ben, 2026-09-10). Triggered per machine by
# its previous config finishing (c9 on naomio, c8 on rml3). First trigger performs
# the one-time switch: stop rml2's whole-config c7, freeze+split its rows into 36
# per-phi seed files, restart rml2 on its slice range. Every trigger then starts
# the calling machine's own range. Idempotent; concurrency-guarded.
#
#   bash grid5/c7_split.sh rml2|naomio|rml3
#
# Ranges (throughput-weighted): naomio 000..140 asc (15) | rml2 150..240 asc (10,
# 10 shards standing order) | rml3 350..250 desc (11).
set -u
cd "$(dirname "$0")/.."
OUT=results/gait_sweep
SEED=$OUT/c7seed
MARKER=$OUT/C7_SPLIT_DONE
LOCK=$OUT/.c7split.lock
PY="$PWD/.sweep_venv/bin/python"
M="${1:?usage: c7_split.sh rml2|naomio|rml3}"
KEY=~/.ssh/id_ed25519_ben

log() { echo "== $(date '+%F %T') c7_split[$M]: $*"; }

# ---------- one-time: freeze rml2 whole-config, split seeds, restart rml2 ----------
if [ ! -f "$MARKER" ]; then
  if mkdir "$LOCK" 2>/dev/null; then
    log "first trigger — freezing rml2 whole-config c7"
    touch $OUT/WATCHDOG_OFF                      # neutralize the c7 cron watchdog for good
    P1=run_mach; P2=grid5_sw
    pkill -f "${P1}ine.sh" 2>/dev/null; pkill -f "${P2}eep.py" 2>/dev/null; sleep 3
    C=$OUT/sweep_grid5_c7_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
    log "splitting $(($(wc -l < $C) - 1)) rows into per-phi seeds"
    mkdir -p "$SEED"
    "$PY" - <<PYEOF
import csv
writers = {}; files = {}
with open("$C") as f:
    rd = csv.reader(f); hdr = next(rd)
    for r in rd:
        try: phi = int(float(r[1]))
        except (ValueError, IndexError): continue
        if phi not in writers:
            fh = open(f"$SEED/seed_phi{phi:03d}.csv", "w", newline=""); files[phi] = fh
            w = csv.writer(fh); w.writerow(hdr); writers[phi] = w
        writers[phi].writerow(r)
for fh in files.values(): fh.close()
print("seeds:", len(writers))
PYEOF
    mv "$C" "$C.wholecfg_archive"                # frozen archive; slices are truth now
    # local (rml2) slice pre-seed for its own range 150..240
    for P in $(seq 150 10 240); do
      T=$(printf "phi%03d" "$P")
      cp "$SEED/seed_${T#phi}.csv" 2>/dev/null "$OUT/sweep_grid5_c7_${T}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv" 2>/dev/null || \
      cp "$SEED/seed_${T}.csv" "$OUT/sweep_grid5_c7_${T}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv"
    done
    echo 14 > $OUT/SHARDS_RML2
    setsid nohup bash grid5/run_slice_range.sh c7 14 150 240 SHARDS_RML2 >> $OUT/c7_slices_rml2.log 2>&1 < /dev/null &
    log "rml2 governed slices 150-240 launched (control=14)"
    touch "$MARKER"; rmdir "$LOCK"
  else
    log "another trigger holds the split lock; waiting"
    while [ ! -f "$MARKER" ]; do sleep 10; done
  fi
fi

# ---------- per-machine start ----------
case "$M" in
  rml2) log "rml2 handled in the one-time step" ;;
  naomio)
    TAR=/tmp/c7seed_naomio.tgz
    tar czf "$TAR" -C "$SEED" $(for P in $(seq 0 10 140); do printf "seed_phi%03d.csv " "$P"; done)
    scp -i $KEY "$TAR" naomio@172.26.35.77:/tmp/ >/dev/null
    ssh -i $KEY naomio@172.26.35.77 'cd ~/pengu && git pull --ff-only >/dev/null 2>&1; cd pengu_mujoco/results/gait_sweep; mkdir -p /tmp/c7s && tar xzf /tmp/c7seed_naomio.tgz -C /tmp/c7s; for f in /tmp/c7s/seed_phi*.csv; do P=$(basename $f .csv); P=${P#seed_}; C=sweep_grid5_c7_${P}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv; [ -f "$C" ] || cp "$f" "$C"; done; echo 30 > SHARDS_NAOMIO; cd ../..; setsid bash grid5/run_slice_range.sh c7 30 0 140 SHARDS_NAOMIO > results/gait_sweep/c7_slices_naomio.log 2>&1 < /dev/null & sleep 15; echo "naomio c7 live: $(pgrep -fc "slice_ph""i.py" || echo 0)"'
    log "naomio range 000-140 started"
    ;;
  rml3)
    TAR=/tmp/c7seed_rml3.tgz
    tar czf "$TAR" -C "$SEED" $(for P in $(seq 250 10 350); do printf "seed_phi%03d.csv " "$P"; done)
    scp -i $KEY "$TAR" rml3@172.24.49.235:/tmp/ >/dev/null
    ssh -i $KEY rml3@172.24.49.235 'cd ~/Documents/ben/pengu && git pull --ff-only >/dev/null 2>&1; cd pengu_mujoco/results/gait_sweep; mkdir -p /tmp/c7s && tar xzf /tmp/c7seed_rml3.tgz -C /tmp/c7s; for f in /tmp/c7s/seed_phi*.csv; do P=$(basename $f .csv); P=${P#seed_}; C=sweep_grid5_c7_${P}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv; [ -f "$C" ] || cp "$f" "$C"; done; echo 14 > SHARDS_RML3; cd ../..; setsid bash grid5/run_slice_range.sh c7 14 350 250 SHARDS_RML3 > results/gait_sweep/c7_slices_rml3.log 2>&1 < /dev/null & sleep 15; echo "rml3 c7 live: $(pgrep -fc "slice_ph""i.py" || echo 0)"'
    log "rml3 range 350-250 started"
    ;;
esac
