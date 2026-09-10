// webpage.ino — the whole interface. There is no terminal any more, so every control the
// tuning session needs is a button here and nothing else is.
//
// Styled after the pengu_k0_com105_* pages: dark cards, big rounded buttons, a status dot.
// What it does NOT inherit from them is the 4 Hz /data poll. Nothing may be transmitted
// while the robot walks -- a radio write inside the control loop perturbs the very period
// the recording exists to measure -- so the state shown here is the reply to the last
// button press, not a live feed. The live numbers come out of /dump afterwards.
//
// What is deliberately NOT on this page: kappa, kp, ki, kd, the torso clamp, the torso
// current limit, the leg P gain, the telemetry rate. Those are decided once and flashed
// (see the block at the top of pengu_tune_wifi.ino). A value that has to be re-picked after
// every power cycle ends up wrong in half the records.

// The Arduino build concatenates the tabs in order: the main sketch, then the others
// alphabetically -- so this file is compiled BEFORE wireless.ino and has to bring in the
// WiFi types itself. Include guards make the second include free.
#include <WiFiNINA.h>

const char WEBPAGE[] PROGMEM = R"HTML(<!doctype html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,user-scalable=no">
<title>Pengu c6</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',-apple-system,system-ui,sans-serif;background:#0f1117;color:#e0e0e0;
     min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:20px;gap:14px}
h1{font-size:1.4rem;color:#fff;margin-top:4px}

.status-bar{display:flex;align-items:center;gap:10px;background:#1a1d27;
            border-radius:10px;padding:10px 16px;width:100%;max-width:500px}
.dot{width:10px;height:10px;border-radius:50%;background:#f44336;flex:none}
.dot.on{background:#4caf50;box-shadow:0 0 6px #4caf50}
#lat{margin-left:auto;font-size:.75rem;color:#555}

.ctrl{display:grid;grid-template-columns:1fr 1fr;gap:12px;width:100%;max-width:500px}
.btn{padding:22px;border:none;border-radius:14px;font-size:1.1rem;font-weight:700;color:#fff;
     cursor:pointer;transition:transform .1s,opacity .1s;-webkit-tap-highlight-color:transparent}
.btn:active{transform:scale(.95);opacity:.8}
.btn-ready{background:#2d6a4f}
.btn-walk{background:#1d6fa4}
.btn-ready.active{background:#4caf50;box-shadow:0 0 12px #4caf5066}
.btn-walk.active{background:#2196f3;box-shadow:0 0 12px #2196f366}

.row{display:flex;gap:8px;margin:6px 0}
.mini{flex:1;padding:12px 6px;border:none;border-radius:10px;font-size:.85rem;font-weight:600;
      background:#37474f;color:#fff;line-height:1.25;cursor:pointer;
      -webkit-tap-highlight-color:transparent}
.mini:active{transform:scale(.95);opacity:.8}
.mini small{font-weight:400;opacity:.75;font-size:.72rem}
.mini.sel{background:#1d6fa4;box-shadow:0 0 10px #2196f366}
.btn-stop{background:#7f2f2f}
.warn{background:#8a5a2b}
a.dl{flex:1;padding:12px 6px;border-radius:10px;background:#26405e;color:#dbeaff;font-size:.85rem;
     font-weight:600;text-align:center;text-decoration:none;line-height:1.25}

.card{background:#1a1d27;border-radius:12px;padding:18px;width:100%;max-width:500px}
.card h2{font-size:.7rem;text-transform:uppercase;letter-spacing:1px;color:#555;margin-bottom:14px}
.knob{display:flex;align-items:center;margin-bottom:6px}
.kl{font-size:.8rem;color:#777;flex:1}
.kv{font-size:1rem;font-weight:700;color:#8fd0ff;font-variant-numeric:tabular-nums;
    text-align:right;min-width:70px}
.kv.over{color:#ff8a6b}

.param-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.pi{text-align:center;background:#12141c;border-radius:8px;padding:10px 6px}
.pl{font-size:.65rem;color:#555;text-transform:uppercase;margin-bottom:4px}
.pv{font-size:1rem;font-weight:600;color:#aaa;font-variant-numeric:tabular-nums}

#st{width:100%;max-width:500px;padding:12px;border-radius:10px;background:#12141c;
    font:12px/1.5 ui-monospace,Menlo,monospace;color:#9fb3c8;white-space:pre-wrap;min-height:3em}
</style></head><body>

<h1>&#128039; Pengu c6</h1>

<div class="status-bar">
  <div class="dot" id="dot"></div>
  <span id="stx">idle</span>
  <span id="lat"></span>
</div>

<div class="ctrl">
  <button class="btn btn-ready" id="b-ready" onclick="c('r')">Ready</button>
  <button class="btn btn-walk"  id="b-walk"  onclick="c('w')">Walk</button>
</div>

<div class="card">
  <div class="row">
    <button class="mini btn-stop" onclick="c('q')">STOP</button>
    <button class="mini btn-stop" onclick="c('0')">AMPS 0</button>
    <button class="mini" onclick="c('i')">re-init</button>
  </div>
  <div class="row">
    <a class="dl" href="/dump" download="pengu.csv">DOWNLOAD RECORDING</a>
  </div>
</div>

<div class="card">
  <h2>Presets &mdash; c6: kappa 2, COM 1.31 &mdash; GRID-6 hardware-sweep champions (cap 354, 56 ms, feedforward torso), robust + clearance filtered</h2>
  <div class="row">
    <button class="mini" id="g-1" onclick="pick('1')">1 &middot; mu 0.12 ice<br>
      <small>1.60/260/115/24/20 &mdash; ff 18.6/209, sim 0.19&ndash;0.44 m/s over all leads, crank 578 OVER</small></button>
    <button class="mini" id="g-3" onclick="pick('3')">3 &middot; torso calib<br>
      <small>1.39/240/80/16/30 &mdash; the bout that never fell (kappa 0)</small></button>
    <button class="mini" id="g-2" onclick="pick('2')">2 &middot; mu 0.45 floor<br>
      <small>1.60/260/125/16/35 &mdash; ff 8.3/75, sim 0.19&ndash;0.25 m/s over all leads, crank 628 OVER</small></button>
  </div>
</div>

<div class="card">
  <h2>Torso</h2>
  <div class="row">
    <button class="mini" id="b-torso" onclick="c('T')">mode: <span id="tl">&mdash;</span><br>
      <small>PID &rarr; HELD &rarr; FF &mdash; FF is timed off the gait, not the IMU</small></button>
    <button class="mini" id="b-probe" onclick="c('P')">plant probe: <span id="pb">off</span><br>
      <small>legs still, torso swept 1&ndash;6 Hz open loop</small></button>
  </div>

  <div class="knob"><span class="kl">ff phase &phi;<br><small>the one number to trim on the robot</small></span><span class="kv" id="v-phi2">&mdash;</span></div>
  <div class="row"><button class="mini" onclick="c('h')">&minus; 5&deg;</button>
                   <button class="mini" onclick="c('H')">+ 5&deg;</button>
                   <button class="mini" onclick="c('Q')">+ 45&deg;</button></div>

  <div class="knob"><span class="kl">ff amplitude</span><span class="kv" id="v-amp">&mdash;</span></div>
  <div class="row"><button class="mini" onclick="c('a')">&minus; 0.5&deg;</button>
                   <button class="mini" onclick="c('A')">+ 0.5&deg;</button></div>

  <div class="knob"><span class="kl">ff gain &nbsp;<small>1.0 = &kappa; 0</small></span><span class="kv" id="v-gain">&mdash;</span></div>
  <div class="row"><button class="mini" onclick="c('g')">&minus; 0.05</button>
                   <button class="mini" onclick="c('G')">+ 0.05</button></div>
</div>

<div class="card">
  <h2>Gait</h2>

  <div class="knob"><span class="kl">leg extension</span><span class="kv" id="v-leg">&mdash;</span></div>
  <div class="row"><button class="mini" onclick="c('j')">&minus; 5&deg;</button>
                   <button class="mini" onclick="c('k')">+ 5&deg;</button>
                   <button class="mini warn" onclick="c('F')">&plusmn; flip sign</button></div>

  <div class="knob"><span class="kl">leg swing</span><span class="kv" id="v-sw">&mdash;</span></div>
  <div class="row"><button class="mini" onclick="c('n')">&minus; 2&deg;</button>
                   <button class="mini" onclick="c('m')">+ 2&deg;</button></div>

  <div class="knob"><span class="kl">hip_phi</span><span class="kv" id="v-phi">&mdash;</span></div>
  <div class="row"><button class="mini" onclick="c(',')">&minus; 10&deg;</button>
                   <button class="mini" onclick="c('.')">+ 10&deg;</button></div>

  <div class="knob"><span class="kl">frequency</span><span class="kv" id="v-f">&mdash;</span></div>
  <div class="row"><button class="mini" onclick="c('[')">&minus; 0.05</button>
                   <button class="mini" onclick="c(']')">+ 0.05</button></div>

  <div class="knob"><span class="kl">hip_off (lean)</span><span class="kv" id="v-off">&mdash;</span></div>
  <div class="row"><button class="mini" onclick="c('o')">&minus; 5&deg;</button>
                   <button class="mini" onclick="c('O')">+ 5&deg;</button></div>
</div>

<div class="card">
  <h2>Rates against the measured 354 deg/s ceiling</h2>
  <div class="param-grid">
    <div class="pi"><div class="pl">crank</div><div class="pv" id="p-cr">&mdash;</div></div>
    <div class="pi"><div class="pl">hip</div><div class="pv" id="p-hp">&mdash;</div></div>
    <div class="pi"><div class="pl">buffered</div><div class="pv" id="p-bf">&mdash;</div></div>
  </div>
</div>

<div id="st">Ready, dial the amplitudes up, Walk. The run is held in the robot's RAM;
press DOWNLOAD after stopping. The buffer is a ring, so an over-long bout keeps its last
seconds -- the part with the fall in it.</div>

<script>
// One request at a time. Every button press is a full HTTP round trip to a board that is
// also running a control loop, so queueing them would pile up latency; a press while one is
// in flight is dropped instead.
var busy = false;

function g(s, k){ var m = s.match(new RegExp(k + "\\s+(-?[0-9.]+)")); return m ? m[1] : null; }

function show(s, ms){
  document.getElementById('st').textContent = s;
  document.getElementById('dot').className = 'dot on';
  document.getElementById('lat').textContent = ms + ' ms';

  var f = {'v-leg':'leg', 'v-sw':'swing', 'v-phi':'phi', 'v-f':'freq', 'v-off':'off',
           'p-cr':'crank', 'p-hp':'hip', 'p-bf':'buffered'};
  for (var id in f){ var v = g(s, f[id]); if (v !== null) document.getElementById(id).textContent = v; }

  // the robot appends OVER when pi*f*A_leg exceeds the measured 354 deg/s crank ceiling
  var over = s.indexOf('OVER') >= 0;
  document.getElementById('v-leg').className = over ? 'kv over' : 'kv';
  document.getElementById('p-cr').style.color = over ? '#ff8a6b' : '#aaa';

  var probing = s.indexOf('PROBE') >= 0;
  document.getElementById('pb').textContent = probing ? 'ON' : 'off';
  document.getElementById('b-probe').className = probing ? 'mini warn' : 'mini';

  var tm = s.indexOf('torso FF') >= 0 ? 'FF'
         : s.indexOf('torso HELD') >= 0 ? 'HELD' : 'PID';
  document.getElementById('tl').textContent = tm;
  document.getElementById('b-torso').className = tm === 'PID' ? 'mini' : 'mini warn';
  var gg = function(k){ var mm = s.match(new RegExp(k + "\\s+(-?[0-9.]+)")); return mm ? mm[1] : null; };
  var ff = {'v-phi2':'fphase', 'v-amp':'famp', 'v-gain':'fgain'};
  for (var q in ff){ var vv = gg(ff[q]); if (vv !== null) document.getElementById(q).textContent = vv; }

  var st = s.indexOf('-> WALK') >= 0 ? 'walk' :
           s.indexOf('-> READY') >= 0 ? 'ready' :
           s.indexOf('-> IDLE') >= 0 ? 'idle' : null;
  if (st) setState(st);
}

function setState(s){
  document.getElementById('stx').textContent =
    s === 'walk' ? 'walking -- radio silent until STOP' : s;
  document.getElementById('b-ready').classList.toggle('active', s === 'ready');
  document.getElementById('b-walk').classList.toggle('active',  s === 'walk');
}

function pick(k){
  c(k);
  ['1','2','3'].forEach(function(x){
    document.getElementById('g-' + x).classList.toggle('sel', x === k);
  });
}

function c(k){
  if (busy) return;
  busy = true;
  var t0 = Date.now();
  fetch('/cmd?key=' + encodeURIComponent(k))
    .then(function(r){ return r.text(); })
    .then(function(t){ busy = false; show(t, Date.now() - t0); })
    .catch(function(){
      busy = false;
      document.getElementById('dot').className = 'dot';
      document.getElementById('st').textContent = 'no reply';
    });
}

window.addEventListener('load', function(){ c('?'); });   // unknown key: just fetches state
</script></body></html>)HTML";

void send_webpage(WiFiClient client) {
  client.print(F("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"));
  const int CHUNK = 256;
  int len = strlen_P(WEBPAGE);
  char buf[CHUNK + 1];
  for (int i = 0; i < len; i += CHUNK) {
    int sz = min(CHUNK, len - i);
    memcpy_P(buf, WEBPAGE + i, sz);
    buf[sz] = 0;
    client.print(buf);
  }
}
