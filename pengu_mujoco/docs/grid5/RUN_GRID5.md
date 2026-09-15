# RUN_GRID5 — operations guide for the round-2 sweep

What GRID-5 is: the 10-config co-design sweep (kappa{0,2} x COM{1.05,1.10,1.20,
1.31,1.40}), staged start + extended metrics, 2,142,720 rows/config. The frozen
protocol and all evidence: `docs/grid5_design.md`. Planning memo:
`docs/grid5/session_2026-08-26_planning_memo.md`. Quick launch card:
`docs/grid5/fleet_launch.md`. Analysis/plotting brief (separate
plot-only session): `docs/grid5/PLOT_GRID5.md`. All sweep code lives in `grid5/` (the GRID-4
pipeline in `physics/` is the untouched backup — never edit it).

## 0. PROTOCOL REVISION grid5-v2 (2026-08-26, late night)

The map is now a PURE DETERMINISTIC sweep (Ben): exact nominal mu, no pose jitter,
no RNG, K=1 — twice-run bit-identical. The first launch (v1, jittered rows) was
killed after ~half a day; its partial CSVs are archived in
`results/gait_sweep/old_jittered_v1/` on each machine and must never be merged
(the manifest protocol tag enforces this). DR happens only at the post-map
champion stage. Deploy commands below are unchanged.
v2.1 (same night): hip_phi restored to the FULL 360-deg circle (the {150..190}
trim's evidence was contaminated by the GRID-4 start protocol); rows/config are
now 2,488,320. v2.2/v2.3 (2026-08-26): leg_amp 75-165 @10, hip_amp 12-32 @4 ->
4,147,200 rows/config (~2.3x GRID-4; 65/8 dropped to save ~22%). Completed v2/v2.1
rows stay valid — machines just pull and restart, and resume adds the new cells.

## 1. Deployment status (relaunched 2026-08-26 under grid5-v2)

| machine | queue | status |
|---|---|---|
| naomio (32 cores) | c4 -> c2 -> c9 | RUNNING v2 (c4, 30 shards; relaunched after the v1 kill) |
| rml2 (16 cores) | c6 -> c7 | RUNNING v2 (c6, 14 shards; v1 measured ~51k rows/h -> ~41 h/config) |
| rml3 (16 cores) | c5 -> c10 | RUNNING v2 (c5, 14 shards) |
| mac | c3 -> c8 | TO DEPLOY (Ben) |
| laptop | c1 | TO DEPLOY (Ben) |

All three running machines were killed and relaunched under grid5-v2 on 2026-08-26
(late night); their v1 partials sit in `results/gait_sweep/old_jittered_v1/`.
Verify any machine is on v2 with:
`grep -m1 "DR=NONE" results/gait_sweep/grid5_<cfg>_run.log` (must print) and
`"protocol": "grid5-v2"` in the CSV's manifest.json.

Phase A = c1..c6 (leading config on every machine + c2 second on naomio);
Phase B = c7..c10 queued behind. Queues advance automatically on `.done`.

## 2. Deploying a machine (mac / laptop — two lines)

From the repo root `pengu_mujoco/` on branch `friction-experiments`:

    git pull --ff-only
    nohup bash grid5/run_machine.sh mac    > results/gait_sweep/machine_mac.log    2>&1 &   # mac
    nohup bash grid5/run_machine.sh laptop > results/gait_sweep/machine_laptop.log 2>&1 &   # laptop

First run auto-builds `.sweep_venv` (mujoco 3.8.x, pinned) if missing — takes a
few minutes, then shards start. Nothing else to configure: shard count defaults
to cores-2; trials are deterministic (grid5-v2, K=1); the manifest is written
automatically.

**If this machine ever ran the v1 (jittered) launch** — mac/laptop normally have
NOT, so skip this — clean up BEFORE relaunching, or resume will silently keep the
contaminated rows:

    touch results/gait_sweep/WATCHDOG_OFF
    pkill -f 'run_machine[.]sh' ; pkill -f 'grid5_sweep[.]py'
    mkdir -p results/gait_sweep/old_jittered_v1
    mv results/gait_sweep/sweep_grid5_c*_* results/gait_sweep/machine_*.log \
       results/gait_sweep/grid5_c*_run.log results/gait_sweep/old_jittered_v1/
    rm -f results/gait_sweep/WATCHDOG_OFF

What the one line does, per config in the queue:
initcsv + `manifest.json` -> N shards (`run_sweep.sh`, resume-safe by axis-tuple)
-> installs the grid5 watchdog (@reboot + 10-min cron; grid5-tagged lines only,
never touches grid4 crontab entries) -> polls `.done` every 5 min, reviving dead
shards -> on `.done`, prints the ship-back line and starts the next config.
Re-running the same line after any crash or reboot is always safe: `.done`
configs are skipped, partial configs resume exactly where every machine left off
(rows are seeded per (cell, mu, repeat) — machine-independent).

## 3. Monitoring

    tail -f results/gait_sweep/machine_<name>.log      # queue log (config-level)
    tail -f results/gait_sweep/grid5_<cfg>_run.log     # shard log (cell progress)
    wc -l results/gait_sweep/sweep_grid5_<cfg>_*.csv   # rows done; full = 4,147,201 incl. header; c6 only = 5,184,001 (v2.4: mu 0.9 on c6 only)

Throughput anchor (measured on rml2, 14 shards): ~51,000 rows/h -> ~41 h per
config; naomio at 30 shards should run roughly 2x that. Watchdog activity:
`results/gait_sweep/watchdog5.log`.

## 4. Stopping / restarting

Stop ON PURPOSE (the watchdog will otherwise revive shards):

    touch results/gait_sweep/WATCHDOG_OFF
    pkill -f 'grid5_sweep[.]py' ; pkill -f 'run_machine.sh'

Restart later: `rm -f results/gait_sweep/WATCHDOG_OFF`, then re-run the machine's
one line. NEVER edit a running .sh (bash reads incrementally — the 18.5 h lesson);
editing a running .py is safe. Do not mix DR_K values into one CSV.

## 5. When a config completes (ship-back)

The queue log prints the exact line; in general:

    awk 'NF' <csv> > t && mv t <csv>
    gzip -kf <csv>
    split -b 90m -d <csv>.gz <csv>.gz.part          # v2.3 gz ~174 MB > GitHub's 100 MB cap
    git add -f <csv>.gz.part* <csv-stem>.manifest.json   # results/ and *.csv are gitignored
    git commit    # message by Ben; CONFIRM THE BRANCH before pushing (friction-experiments)

Reassemble on any machine:  cat <csv>.gz.part* > <csv>.gz && gunzip -k <csv>.gz

Ship the manifest WITH the CSV — grid5 analysis tools refuse a CSV whose manifest
does not match (protocol, K, slip constants, mujoco version). Fleet rules apply:
commit local run data before pulling; announce any `results/` layout change in a
memo BEFORE pushing it.

## 6. Troubleshooting

- "manifest missing — run initcsv first": the CSV exists but its manifest.json
  does not (e.g. copied by hand). Run `CONFIG=<cfg> python grid5/grid5_sweep.py
  initcsv` from `grid5/`, or re-run the machine line.
- "MANIFEST MISMATCH ... refusing to write": the process settings (protocol, K,
  slip constants, mujoco version) differ from the artifact's manifest. Do NOT
  force — fix the environment (wrong mujoco, or a stale v1 CSV: archive it).
- CSV with no header: resume silently recovers 0 rows and re-runs everything —
  the launcher and watchdog refuse to run on a header-less CSV; repair per
  `docs/grid4_fleet_memo.md` pre-flight 4 (same trap as GRID-4).
- Two machines on ONE config: safe ONLY with disjoint shard ids (same N_SHARDS,
  different SHARD_ID sets, both after one initcsv). Default setup = one config
  per machine; prefer that.
- Reassigning queues (e.g. laptop finished c1): edit the queue table at the top
  of `grid5/run_machine.sh`, commit, pull on the target machine, re-run its line.
- Shard RSS climbing toward ~1 GB each / swap filling / kernel oom_kill in dmesg:
  you are running a pre-2026-08-31 build. The old tuple-set resume loaded EVERY
  done row into every shard (~1.3 GB/shard at 3M rows; OOM-killed naomio's shards
  30 x 1.3 GB on 32 GB). Pull — the bitmap resume (1 bit/row, ~0.65 MB) replaced
  it. General rule: per-shard resume state must NOT scale with map size.
  First OOM in project history; prior "no OOM" forensic signatures (machines B/C,
  the bash-edit incident) were all ruled-out cases.

## 6b. Demo / render output convention (Ben, 2026-08-30)

All demos rendered FROM SWEEP DATA (champion clips, per-config showcase videos,
trace figures) go to `results/grid5_report/<cfg>/demos/` — mirroring the
grid4_report layout. `results/grid5_probes/` is reserved for math/validation
probe outputs only (e.g. the IMU frame demo). Demo videos use TWO camera angles
(side + front, hstacked) like grid4_demos.py, and champion clips should ship with
a torso-roll trace figure when kappa != 0.

## 6c. rml2 shard governor (live control, Ben 2026-09-02)

rml2 runs c2 slices through `grid5/run_c2_governed.sh`. Shard count is LIVE-tunable
via a one-integer control file — no restarts, no downtime beyond a <1 s slice reload:

    echo 14 > results/gait_sweep/SHARDS_RML2    # desktop free: full speed
    echo 8  > results/gait_sweep/SHARDS_RML2    # someone working: 60%
    echo 0  > results/gait_sweep/SHARDS_RML2    # pause entirely

The governor reconciles every 20 s (kill + bitmap-resume relaunch — lossless, rows
flush per line and slice CSVs are small) and doubles as the watchdog (revives dead
shards). Typical schedule: 14 from 04:00, drop to 8 when the desktop is needed
(~13:00) — but it's manual by design; just say the number (or write the file).

## 7. After the maps (not yet started)

Per config: the champion DR stage (jittered repeats on selected rows; exact design
fixed AFTER the map completes — `grid5/topup_k.py`'s v1 merge is guarded out until
then), then the selection chain (three champion tracks T-speed/T-cot/T-slip,
independent-seed confirmation, neighborhood fine scan) via `grid5/grid5_select.py`
(to be written before analysis begins — see grid5_design.md). Robust-region reporting is four-tier: surv-only / pass / strict
heading>=0.9 / clean-pass slip<=0.05, all recomputable from the saved records.

## CAMPAIGN COMPLETE — 2026-09-12

All ten GRID-5 maps are shipped to `friction-experiments` as canonical
`sweep_grid5_cN_*.csv.gz.part*` + manifest (reassemble: `cat parts > gz`).
Final three ship commits: c8 267c282 (09-10), c9 90ba108 (09-10), c7 608e420 (09-12).
Every map passed the full battery at ship time (row/marginal/gate/fall-phase checks,
0 violations). Battery mu-summaries from the final three:

  c7 (k0 com1.10): surv 93.4/90.9/72.0/55.8 %  pass 11.2/20.1/12.4/6.7 %   (mu .1/.3/.5/.7)
  c8 (k0 com1.40): surv 32.7/24.7/12.0/ 9.2 %  pass  5.2/ 5.1/ 1.0/0.4 %
  c9 (k2 com1.10): surv 68.1/70.9/65.8/58.5 %  pass 19.8/23.2/10.4/7.4 %

c7 was assembled from 36 per-phi slices across three machines (ownership + handoff
history: `results/gait_sweep/c7_OWNERSHIP.txt`; provenance in the canonical manifest).
c8's manifest records the mac partial seeding + the phi350 dedup/gap-fill.

Ops lessons added this week:
- A slice "complete" by LINE count can still be short on UNIQUE keys (seed injected
  while shards ran duplicates rows). Always audit `cut -d, -f1-6 | sort -u | wc -l`
  == 115,200 per slice before merging. (Bit c8 phi350; re-checked everywhere since.)
- Never let two governors run on one machine: `stopall` pkills every slice_phi on the
  host, so concurrent governors fight (kill/relaunch livelock) and can duplicate rows.
  Chain a waiter (`while pgrep governor; do sleep; done; exec new-governor`) instead.
- `pgrep -c -f X || echo 0` emits "0\n0" when nothing matches (pgrep prints 0 AND
  fails) — numeric tests then silently fail. Drop the `|| echo 0`.
- SHARDS_* control files are shared per machine: a paused (=0) file left by one range
  freezes the next range's governor. Check `cat SHARDS_*` when shards mysteriously
  stop; 0 may also be a deliberate manual pause (respect it — ask before overriding).
- rml3's address changed after its 09-11 reboot: now rml3@172.24.49.235.
