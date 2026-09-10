// pengu_hw_c3.ino — pengu_tune_wifi with the c3 cap-only-sweep champions as presets 1 / 2, PID torso.
//
// The USB socket on the old board broke when the robot fell and pulled the cable, so there
// is no serial interface here at all: nothing is read from Serial, every control is a
// button at http://192.168.4.1, and the run is recorded into RAM and downloaded
// afterwards instead of being streamed. Nothing is transmitted while the robot walks:
// a radio write inside the control loop would perturb the very period this data is
// meant to measure. Serial is a boot log only (MIRROR_SERIAL 0 drops it).
//
// Both amplitudes boot at ZERO. Press WALK and the robot takes its lean and then stands
// there; nothing oscillates until an amplitude is dialled up. Add one mechanism at a time
// and watch which combination cooperates.
//
//   leg extension amplitude   +-5 deg    crank, unipolar 0..amp
//   leg swing amplitude       +-2 deg    hip, half-rectified
//   hip_phi                   +-10 deg   hip phase RELATIVE TO the crank
//   frequency                 +-0.05 Hz  bumpless, safe to change mid-walk
//   hip_off                   +-5 deg    forward lean
//
// c3: kappa = 0, COM ratio 1.31  (sim model pengu1_31). SET THE BALLAST FOR 1.31 BEFORE WALKING.
// Presets 1 and 2 are the GRID-6 CAP-ONLY sweep champions (2026-09-09, psc/hw_cap.slurm): 354 deg/s
// crank cap, kappa PID torso, NO torso lag and NO feedforward in the model; selected among cells
// that walk with every freq +-0.05 / hip_phi +-10 neighbour (nbhd pass >= 0.8) AND clear the floor
// by >= 10 mm at the worst cycle, then ranked on whole-body-COM net speed.
//   1   1.25 / 280 / 125 / 24 / 20   mu 0.12 (ice)   sim v 0.142 m/s, clear 13 mm, torso rms 2.5, clamp 2% of the time
//   2   1.65 / 310 / 130 / 20 / 20   mu 0.45 (floor) sim v 0.248 m/s, clear 39 mm, torso rms 5.8, clamp 53% of the time
//   3   1.39 / 240 /  80 / 16 / 30  the torso calibration gait (kappa 0, feedforward)
//   runners-up, mu 0.12:
//        1.35 / 290 / 115 / 24 / 20   v 0.139  clear 10 mm  torso rms 2.6  clamp 4%
//        1.25 / 280 / 120 / 24 / 20   v 0.138  clear 11 mm  torso rms 2.4  clamp 0%
//   runners-up, mu 0.45:
//        1.65 / 320 / 115 / 20 / 20   v 0.244  clear 32 mm  torso rms 5.7  clamp 53%
//        1.65 / 320 / 120 / 20 / 20   v 0.242  clear 36 mm  torso rms 5.7  clamp 52%
// BOOTS INTO THE PID TORSO (torso_mode 0), which is what the sweep modelled. Two things the sweep
// did not have: (a) the 56 ms IMU/servo lag -- the model's loop has none, the robot's does; the
// firmware PID runs KP 0.5 (measured optimum here) where the model runs 2.0, same equilibrium
// J* = (kappa-1)*axis, slower convergence. (b) the clamp: model 25 deg, robot 25 (stop at +27.8/-31).
// T cycles PID -> HELD -> FF; the FF numbers below are the kappa-0 calibration, NOT fitted to these gaits.
//
// Two measured constraints this sketch exists to respect:
//
// 1. CRANK VELOCITY CEILING 354 +- 4 deg/s. Twelve measurements on 2026-08-30, air and
//    ground pooled, demands from 424 to 613 deg/s, two frequencies: every one executed at
//    354. Below 380 the servo tracks at 0.99 with a constant 19-25 ms lag; above it the
//    amplitude ratio collapses (0.93 -> 0.81 -> 0.69) and the lag grows (39 -> 63 -> 88 ms).
//    Peak crank rate is pi*f*A_leg, so the executable region is  f * A_leg <= 113.
//    Every gait flashed before that measurement sat above the ceiling, which means the
//    robot was never running the gait the sweep selected -- it ran a clipped, delayed one.
//
// 2. hip_off 50. Applying the ceiling to the GRID-4 c1 sweep at mu=0.5 leaves 2804 of the
//    55430 passing cells (5%), and almost all of them sit at hip_off 50 -- a lean far past
//    anything tried so far. The best executable cell is 1.04 / 240 / 85 / 28 / 50 at
//    net 0.1223 (crank 278, hip 183, both well inside). hip_phi 0, the fastest region in
//    simulation, does not survive the ceiling at all: it needs f 1.7-2.0 with A_leg 125.
//
// The gait is unchanged from the sweep's definition:
//   crank_L = 0.5*A_leg*(1 + sin(p))              crank_R = 0.5*A_leg*(1 + sin(p + pi))
//   hip_L   = off + A_hip*max(0, sin(p+pi+PHI))   hip_R   = off + A_hip*max(0, sin(p+PHI))

// rec_dump() below takes a WiFiClient, and the IDE concatenates the main sketch ahead of
// wireless.ino where the radio actually lives -- so the type has to be visible here too.
#include <WiFiNINA.h>
#include <DynamixelShield.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

using namespace ControlTableItem;
#define DEBUG_SERIAL Serial
#define MIRROR_SERIAL 1        // boot log only; nothing is ever READ from Serial

// =========================== SET THESE BEFORE UPLOADING ===========================
const float    KAPPA           = 0.0f;   // torso target = kappa * hip-axis roll. 0 = hold level
const float    KP              = 0.5f;   // measured optimum; 1.0 marginal, 2.0 diverges here
const float    KI              = 0.1f;   // contributes under 1 deg on this robot; near inert
const float    KD              = 0.0f;   // [s] on -d(roll)/dt. 0.05 would match the whole KP term
const float    FF_LP_HZ        = 2.0f;   // low-pass on the FEEDFORWARD path only; 0 = off.
// Why this exists. With KAPPA=0 the law expands to  J_cmd = +1.00*J_meas - 1.50*imu_roll:
// the (KAPPA-1)*axis feedforward carries a +1 on the torso's OWN measured angle, because
// axis is reconstructed as (imu_roll - J). Standing still with both gait amplitudes at
// zero, the robot then sat in a 3.1-3.3 Hz limit cycle (pengu-3, pengu-4, 2026-09-02):
// +-23 deg of torso, +-16 deg of body roll, 1300 mA, and opening the loop with 'T' stopped
// it dead (roll rms 8.89 -> 1.30, current 1000 -> 60 mA). Measured round the loop from the
// robot's own S and B the return ratio was 1.006 at 0.7 deg -- the oscillation condition,
// hit to within 1% and 1 deg on three independent segments.
// Lowering KP does not help: the +1*J term is not scaled by KP, and even KP=0 leaves the
// return ratio at 0.83-1.04. Adding KD makes it worse (1.02 at 6 Hz). What works is taking
// the gain out of the SELF term at 3 Hz while keeping it at DC and at the gait frequency,
// so only the feedforward is filtered -- never imu_roll itself, since lag on the error path
// is the disease rather than the cure. At 2 Hz the worst return ratio over 1.5-6 Hz falls
// to 0.38 (a 2.6x margin) while 80% of the feedforward survives at a 1.5 Hz gait, arriving
// 37 deg late. 1.5 Hz gives 3.3x and keeps 71%; 3.0 Hz gives 1.8x and keeps 89%.
float          axis_lp = 0.0f;
const float    TORSO_CLAMP_DEG = 25.0f;  // mechanical stop measured +27.8 / -31.0 (torso_rom)
const uint16_t TORSO_CURRENT_MA= 3210;   // register maximum = no cap. Under 830 mA to bite
const uint16_t LEG_PGAIN       = 1600;   // POSITION_P_GAIN on hips+cranks; torso left alone
const uint16_t TORSO_PGAIN     = 0;      // 0 = leave the torso servo at whatever it holds.
                                         // Its current draw carries 21-66% of its energy in
                                         // 2.5-4.5 Hz at every drive frequency tried, which
                                         // is the inner loop buzzing, not the kappa loop --
                                         // lowering this is the untested direction.
const float    T_RAMP = 4.0f, T_SETTLE = 6.0f, T_BLEND = 4.0f;   // staged start [s]
const float    HIP_REST_DEG    = 10.0f;  // rest lean; the hip_off ramp starts from here
const uint16_t TEL_MS          = 20;     // telemetry period -> 50 Hz, Nyquist 25 Hz. At 20 Hz
                                         // the 2.5-4.5 Hz peak could not be told from an
                                         // alias of something faster; this settles it.
const int      REC_MAX         = 570;    // x22 B = 12540 B. 11.4 s at 50 Hz, 22.8 s at
                                         // 25 Hz. Up here rather than beside the buffer
                                         // because the state line quotes it, and the IDE
                                         // prototypes functions but not constants.
const float STEP_LEG = 5.0f, STEP_HIP = 2.0f, STEP_PHI = 10.0f;
const float STEP_FREQ = 0.05f, STEP_OFF = 5.0f;
const float FREQ_MIN = 0.30f, FREQ_MAX = 2.60f;
// ==================================================================================

// live, on buttons. Amplitudes start at zero; the lean starts at the sweep's 50.
float p_legFreq = 1.25f;      // [Hz]  the executable champion, to the 0.05 step grid
float p_legAmp  = 0.0f;       // [deg] crank amplitude
float p_hipAmp  = 0.0f;       // [deg] hip swing amplitude
float p_hipPhi  = 280.0f;     // [deg] hip phase relative to the crank
float p_hipOff  = 20.0f;      // [deg] forward lean

// Changing frequency mid-walk is not free: phase = 2*pi*f*(t - t0), so raising f at
// t - t0 = 13 s by 0.05 jumps the phase by 0.65 of a cycle and kicks the robot. setFreq()
// absorbs the difference into this offset; at a constant frequency it stays put.
float gait_phi_off = 0.0f;

const uint8_t XM_LEFT_SLIDE = 4, XM_RIGHT_SLIDE = 3;
const uint8_t XM_LEFT_HIP   = 2, XM_RIGHT_HIP   = 1, XM_TORSO_ROLL = 0;
const uint8_t MOTOR_IDS[] = {XM_LEFT_SLIDE, XM_RIGHT_SLIDE, XM_LEFT_HIP, XM_RIGHT_HIP, XM_TORSO_ROLL};
const int     MOTOR_COUNT = 5;
const uint8_t LEG_IDS[4]  = {XM_RIGHT_HIP, XM_LEFT_HIP, XM_RIGHT_SLIDE, XM_LEFT_SLIDE};

DynamixelShield  dxl;
Adafruit_BNO055  bno = Adafruit_BNO055(55, 0x28);

enum RobotState { STATE_IDLE, STATE_READY_SLIDE, STATE_READY_HIP, STATE_READY_TORSO, STATE_WALK };
RobotState robot_state = STATE_IDLE;

float         imu_roll = 0, imu_pitch = 0;
float         grav_x = 0, grav_y = 0, grav_z = 0;
float         roll_rate = 0.0f;          // EMA of d(imu_roll)/dt [deg/s], feeds the D term
float         home_deg[MOTOR_COUNT];
unsigned long walk_start_ms = 0;
float         torso_iErr = 0.0f;
// 'T' opens the torso loop: the kappa PID is not run and the torso simply holds its home
// angle. With both gait amplitudes at zero the loop is then the ONLY thing left moving, so
// this button is what separates "the robot rocks because of the feedback" from "the robot
// rocks whatever the controller does". On 2026-09-02 with amplitudes 0 and the loop CLOSED
// the robot oscillated at 3.11 Hz, +-23 deg of torso and +-16 deg of body roll, drawing
// +-1300 mA -- while the same configuration in simulation is motionless to 0.00 deg.
// No extra column is needed to tell the two apart in a dump: with the loop open goal_torso
// is identically 0.00 for the whole bout, which never happens when it is closed.
// 0 = the kappa PID, 1 = held at home, 2 = phase-locked feedforward.
//
// The PID has the right sign and the wrong timing. Measured on this robot (pengu-A, -B,
// -10): the torso joint reaches its extreme 56 ms after the hip axis reaches its own, by
// which time the axis is already coming back in 76-90% of events, and corr(dJ, d_axis)
// peaks at -0.94 at exactly -56 ms. Pushing the lower body the way it is already going is
// what turns a 21 deg peak-to-peak roll (loop held) into 67 (loop closed), and the torso
// motor is half the robot's mass, so the reaction lands squarely on the legs.
//
// Mode 2 removes the measurement from the loop entirely. The disturbance is periodic and
// the board generates the period itself, so the torso is driven straight off the gait
// phase: no IMU in the path, no 55 ms to wait for. Only the servo's own lag remains, and
// that is cancelled by leading the phase rather than by closing a faster loop.
// In the model, with 56 ms of servo lag applied to both: PID torso_roll_rms 4.74 and the
// axis WORSE at 6.49; feedforward 0.59 with the axis at 5.01. With no lag the ranking
// reverses (0.67 vs 2.83), which is the world the sweep was built in.
// Boots into the feedforward. Calibrated on the robot 2026-09-02 over ten bouts at
// 1.39/240/80/16/30: with the torso HELD the roll was 7.35 deg rms and the bout completed;
// with the kappa PID it was 34.09 and the robot went down at 11.5 s with the torso command
// on its clamp 32% of the time; with the feedforward at phi 119 / A 7.5 it was 1.46 and all
// 21 cycles walked. The phi scan bottoms out between 114 and 119 -- 94, 129, 144 and 159
// read 4.83, 3.03, 4.89, 5.34 -- and the amplitude between 7.5 and 9.0, with 6.0 at 2.32.
// Splitting each bout into thirds puts the run-to-run floor at 0.44 deg, so phi 114 (1.54)
// and A 9.0 (1.59) are NOT distinguishable from the value below; the region is established,
// the exact point is not.
//
// THIS CALIBRATION IS FOR KAPPA = 0 ONLY. ff_gain scales the whole correction, so
// gain 1.0 is kappa 0 and gain 0 is the torso held; other kappa values need their own
// amplitude and phase and none has been measured.
uint8_t       torso_mode = 0;   // PID: the cap-only sweep modelled the kappa loop, not the feedforward
const uint8_t TM_PID = 0, TM_HELD = 1, TM_FF = 2;

// Feedforward parameters. A_t comes from a loop-HELD recording: it is the amplitude of the
// hip-axis roll the torso has to cancel, 7.49 deg on pengu-12 against 7.55 in the model.
// PHI is the one number that must be trimmed on the robot -- the model's optimum sits
// 47.5 deg ahead of the naive value, and only 28 of that is the servo lag; the rest is the
// plant's own phase, which is not transferable. Sweep it with the buttons and watch the
// roll. GAIN scales the whole correction: 1.0 is kappa = 0, and a smaller value is a
// partial correction, i.e. a point between kappa 0 and 1.
float         ff_amp  = 7.5f;    // [deg]  measured on the robot, see above
float         ff_phi  = 119.0f;  // [deg] relative to the gait phase. The naive value that
                                 // cancels the measured axis roll is 75.9; the best on the
                                 // robot is 43 deg past it, which the model predicted would
                                 // be 45-70 past. Only the offset had to be measured.
float         ff_gain = 1.0f;

// 'P' runs a plant identification instead of a gait: the legs hold their rest lean, the
// kappa loop is not used, and the torso is driven open-loop by a sine whose frequency steps
// through PROBE_HZ. One recording yields B(f) = torso world roll / torso joint angle -- the
// transfer the kappa loop closes around, measured on the robot rather than assumed. At
// 3.11 Hz the closed-loop records give |B| 0.789 at -64.6 deg while the model gives 0.584
// at -40.0, and that gap is what decides whether the loop oscillates at all. If the phase
// of B turns out linear in frequency, its slope is a pure transport delay and the remainder
// is mechanics; only the second half is something the simulation can be asked to reproduce.
// No bookkeeping is needed in the record: goal_torso IS the drive, so the analysis reads
// each segment's frequency straight out of the logged command.
// pengu-5 (2026-09-02) had every one of these but the last scrolled out of the ring: the
// schedule ran 14.4 s, the buffer holds 570 records and the loop was going at 43.5 Hz, so
// only the final 13.1 s survived and all of it was the clamped last frequency. Hence a
// shorter total AND an automatic stop at the end of the sweep -- the bout now ends itself,
// so the buffer holds exactly the probe and nothing has to be timed by hand.
// Low frequencies get longer segments: what matters is cycles per segment, not seconds.
const float    PROBE_HZ[]      = {1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.5f, 6.0f};
const float    PROBE_SEC[]     = {2.0f, 1.6f, 1.4f, 1.3f, 1.2f, 1.2f, 1.2f};  // 9.9 s total
const int      PROBE_N         = 7;
const float    PROBE_AMP_DEG   = 8.0f;   // the drive amplitude the simulation sweep used
bool           probe_on    = false;
float          probe_phase = 0.0f;       // integrated, so a frequency step cannot jump it
float         loop_dt_ema = 0.0f;
unsigned long loop_count = 0, loop_hz = 0;
bool          motors_ok = false;

const float READY_STEP = 1.0f, READY_STEP_TORSO = 1.5f, ARRIVE_THRESH = 1.0f;


// from wireless.ino
void  begin_wifi();
void  update_wifi();
void  set_status_rgb(int state);
int   state_str(char *b);    // both defined below; wireless.ino calls them from
void  apply_cmd(char cmd);   // inside the /cmd handler


// -------------------------------------------------------------- status text (log only)
// Status text goes to the serial mirror only. It is never sent over the air: the page gets
// the state as the reply to its own button press, and the recording is fetched afterwards.
void tx(const char *s) {
#if MIRROR_SERIAL
  DEBUG_SERIAL.println(s);
#else
  (void)s;
#endif
}

// Fixed-point float formatter. snprintf("%f") is not linked in by default on the SAMD core,
// and String would fragment the heap inside a control loop, so numbers are built by hand.
static char *fmt(char *p, float v, uint8_t dp) {
  if (v < 0) { *p++ = '-'; v = -v; }
  long scale = 1;
  for (uint8_t i = 0; i < dp; i++) scale *= 10;
  long n  = (long)(v * scale + 0.5f);
  p += sprintf(p, "%ld", n / scale);
  if (dp) {
    *p++ = '.';
    for (long d = scale / 10; d > 0; d /= 10) *p++ = '0' + (char)((n / d) % 10);
  }
  return p;
}
static char *fmtc(char *p, float v, uint8_t dp) { p = fmt(p, v, dp); *p++ = ','; return p; }


// ------------------------------------------------------------------ small helpers
int idxOf(uint8_t id) {
  for (int i = 0; i < MOTOR_COUNT; i++) if (MOTOR_IDS[i] == id) return i;
  return -1;
}
float shortestDelta(float cur, float tgt) {
  float d = tgt - cur;
  while (d >  180.0f) d -= 360.0f;
  while (d < -180.0f) d += 360.0f;
  return d;
}
float zeroExtended(float cur) {       // extended-coord value of physical 0 nearest to cur
  float phys = fmod(cur, 360.0f); if (phys < 0) phys += 360.0f;
  return cur - ((phys > 180.0f) ? phys - 360.0f : phys);
}
void stepMotorToward(uint8_t id, float target, float step) {
  float cur = dxl.getPresentPosition(id, UNIT_DEGREE);
  float phys = fmod(cur, 360.0f); if (phys < 0) phys += 360.0f;
  float d = shortestDelta(phys, target);
  dxl.setGoalPosition(id, cur + ((fabsf(d) <= step) ? d : (d > 0 ? step : -step)), UNIT_DEGREE);
}
bool arrivedAt(uint8_t id, float target) {
  float phys = fmod(dxl.getPresentPosition(id, UNIT_DEGREE), 360.0f);
  if (phys < 0) phys += 360.0f;
  return fabsf(shortestDelta(phys, target)) < ARRIVE_THRESH;
}

void setFreq(float f) {
  f = constrain(f, FREQ_MIN, FREQ_MAX);
  if (robot_state == STATE_WALK) {
    float tw = (millis() - walk_start_ms) / 1000.0f - T_RAMP - T_SETTLE;
    if (tw > 0.0f) gait_phi_off += 2.0f * PI * (p_legFreq - f) * tw;   // bumpless
  }
  p_legFreq = f;
}

// One place builds the state line: it is printed to the log AND returned as the reply to
// every button press, so the page can show what the robot currently has without a second
// endpoint and without the browser ever seeing the telemetry stream.
int state_str(char *b) {
  char *p = b;
  p += sprintf(p, "# freq ");   p = fmt(p, p_legFreq, 2);
  p += sprintf(p, "  leg ");    p = fmt(p, p_legAmp, 0);
  p += sprintf(p, "  swing ");  p = fmt(p, p_hipAmp, 0);
  p += sprintf(p, "  phi ");    p = fmt(p, p_hipPhi, 0);
  p += sprintf(p, "  off ");    p = fmt(p, p_hipOff, 0);
  // peak joint rates against the measured 354 deg/s ceiling
  p += sprintf(p, "   crank "); p = fmt(p, PI * p_legFreq * fabsf(p_legAmp), 0);
  p += sprintf(p, "/354  hip "); p = fmt(p, 2.0f * PI * p_legFreq * p_hipAmp, 0);
  p += sprintf(p, "/354%s", (PI * p_legFreq * fabsf(p_legAmp) > 354.0f) ? "  OVER" : "");
  p += sprintf(p, "   torso %s",
               torso_mode == TM_PID ? "PID" : torso_mode == TM_HELD ? "HELD" : "FF");
  if (torso_mode == TM_FF) {
    p += sprintf(p, " famp ");   p = fmt(p, ff_amp, 1);
    p += sprintf(p, " fphase "); p = fmt(p, ff_phi, 0);
    p += sprintf(p, " fgain ");  p = fmt(p, ff_gain, 2);
  }
  if (probe_on) p += sprintf(p, "   PROBE 7f/9.9s, auto-stops");
  p += sprintf(p, "   ffLP "); p = fmt(p, FF_LP_HZ, 1); p += sprintf(p, "Hz");
  p += sprintf(p, "   buffered ");  p = fmt(p, rec_seconds(), 1);
  p += sprintf(p, "s of %d", (int)(REC_MAX * TEL_MS / 1000));
  p += sprintf(p, "s   loop %luHz", loop_hz);
  *p = 0;
  return (int)(p - b);
}

void announce() {
  // A change made mid-bout has to be timestamped into the recording, or the dumped rows
  // would carry the wrong parameters from that point on.
  if (robot_state == STATE_WALK) rec_event(millis() - walk_start_ms);
  char b[192];
  state_str(b);
  tx(b);
}


// ------------------------------------------------------------------ bring-up
// Split out so a button can redo it: the board is powered over USB/battery and the motors
// are not, so if the 12 V comes on after boot every torqueOn is lost. That produced a whole
// session of "READY prints but nothing moves" on 2026-08-30.
bool initMotors() {
  char b[128];
  bool all_ok = true;
  dxl.begin(1000000);
  dxl.setPortProtocolVersion(2.0);
  for (int i = 0; i < MOTOR_COUNT; i++) {
    uint8_t id = MOTOR_IDS[i];
    if (!dxl.ping(id)) {
      all_ok = false;
      sprintf(b, "No response ID %d   <-- is the 12 V on?", id); tx(b);
      continue;
    }
    float boot_deg = dxl.getPresentPosition(id, UNIT_DEGREE);
    home_deg[i] = zeroExtended(boot_deg);
    dxl.torqueOff(id);
    // EXTENDED position accepts negative / >360 goals; plain OP_POSITION silently rejects
    // them. The torso runs current-based position so its torque ceiling is settable, and
    // mode 5 is still multi-turn, so the goals below are unchanged.
    dxl.setOperatingMode(id, (id == XM_TORSO_ROLL) ? OP_CURRENT_BASED_POSITION
                                                   : OP_EXTENDED_POSITION);
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 0);       // no slew limit
    dxl.writeControlTableItem(PROFILE_ACCELERATION, id, 0);
    dxl.torqueOn(id);
    if (id == XM_TORSO_ROLL) {
      dxl.writeControlTableItem(GOAL_CURRENT, id, (uint16_t)(TORSO_CURRENT_MA / 2.69f));
      if (TORSO_PGAIN) dxl.writeControlTableItem(POSITION_P_GAIN, id, TORSO_PGAIN);
    }
    dxl.setGoalPosition(id, boot_deg, UNIT_DEGREE);           // hold where it is; no snap
    int32_t tq = dxl.readControlTableItem(TORQUE_ENABLE, id);
    if (tq != 1) all_ok = false;
  }
  // This used to be written only by a key nobody pressed while the log printed 1600
  // regardless, so every record before 2026-08-30 ran at the servos' stored gain. Write it,
  // then READ IT BACK, so the number reported is the number in force.
  for (int i = 0; i < 4; i++) dxl.writeControlTableItem(POSITION_P_GAIN, LEG_IDS[i], LEG_PGAIN);

  // What actually caps the leg speed. PROFILE_VELOCITY is set to 0 above, and on the X
  // series that means "no profile -- use VELOCITY_LIMIT", which has never been read here.
  // The bench measurement says 354 +- 4 deg/s over twelve points, and an XM430-W350 turns
  // 276 deg/s no-load at 12 V and about 340 at 15, so 354 looks physical. But the default
  // VELOCITY_LIMIT is 350 raw = 481 deg/s, so if the register reads lower than the bench
  // number the ceiling is a setting rather than the motor, and raising it is free.
  // 1 raw unit = 0.229 rev/min = 1.374 deg/s.
  {
    char b[192], *p = b;
    p += sprintf(p, "# vel_limit raw/degs:");
    for (int i = 0; i < 4; i++) {
      long v = dxl.readControlTableItem(VELOCITY_LIMIT, LEG_IDS[i]);
      p += sprintf(p, " %ld=", (long)LEG_IDS[i]);
      p = fmt(p, (float)v, 0);
      p += sprintf(p, "/");
      p = fmt(p, v * 1.374f, 0);
    }
    p += sprintf(p, "  supply ");
    p = fmt(p, dxl.readControlTableItem(PRESENT_INPUT_VOLTAGE, LEG_IDS[0]) / 10.0f, 1);
    p += sprintf(p, "V");
    *p = 0; tx(b);
  }
  sprintf(b, "# leg POSITION_P_GAIN read back: %ld %ld %ld %ld  (asked %u)",
          (long)dxl.readControlTableItem(POSITION_P_GAIN, LEG_IDS[0]),
          (long)dxl.readControlTableItem(POSITION_P_GAIN, LEG_IDS[1]),
          (long)dxl.readControlTableItem(POSITION_P_GAIN, LEG_IDS[2]),
          (long)dxl.readControlTableItem(POSITION_P_GAIN, LEG_IDS[3]), LEG_PGAIN);
  tx(b);
  tx(all_ok ? "# all 5 motors powered and holding"
            : "# NOT ALL MOTORS ARE HOLDING -- power the 12 V, then press RE-INIT");
  motors_ok = all_ok;
  return all_ok;
}

void setup() {
#if MIRROR_SERIAL
  DEBUG_SERIAL.begin(115200);
  unsigned long t0 = millis();
  while (!DEBUG_SERIAL && millis() - t0 < 1500);   // never block: there may be no cable
#endif
  begin_wifi();
  initMotors();
  if (!bno.begin()) tx("ERROR: BNO055 not detected.");
  else bno.setExtCrystalUse(true);
  tx("# pengu_tune_wifi ready. Join the PENGU network, open http://192.168.4.1");
  announce();
}



// ------------------------------------------------------------- the on-board recorder
// Nothing is transmitted while walking. Samples go into a RAM ring buffer and the whole
// run is fetched afterwards over HTTP, so the control loop never waits on the radio.
//
// Text is far too fat to buffer -- a CSV row is about 200 bytes, which would be four
// seconds of RAM. These records are 21 bytes. What is stored is only what cannot be
// recomputed: every MEASURED quantity, plus the gait `phase` itself. The four commanded
// leg positions are exact functions of phase and the parameters, so rec_dump() rebuilds
// them from the same expressions run_walk() uses -- keeping one copy of the gait maths
// rather than a second one in the analysis script that would silently drift.
//
// The buffer is a RING: when it fills it overwrites the oldest sample. A bout that outlives
// the buffer therefore keeps its LAST N seconds, which is the part containing the fall.
struct __attribute__((packed)) Rec {
  uint16_t t_ms;        // since WALK; wraps at 65.5 s, longer than any bout so far
  uint16_t phase_q;     // gait phase mod 2pi, mapped onto 0..65535
  int16_t  slL, slR;    // measured crank angles, deg x100
  int16_t  hpL, hpR;    // measured hip angles, deg x100
  int16_t  torso;       // measured torso joint, deg x100
  int16_t  gtorso;      // COMMANDED torso, deg x100 -- feedback, so not recomputable
  int16_t  roll, pitch; // IMU, deg x100
  int8_t   mA20;        // torso current / 20 mA, +-2540 mA
  int16_t  gx, gy, gz;  // BNO gravity vector in body axes, m/s^2 x1000 (+-9810 fits).
                        // The Euler pitch wraps through +-180 as soon as the roll gets
                        // large, which is exactly when the interesting gaits run: pengu-A,
                        // -B and -11 all came back with that channel unusable, so their
                        // backward falls could not be counted at all. Gravity never wraps
                        // and carries both tilts. The raw components are logged rather
                        // than an angle because the board does not know how the sensor is
                        // mounted relative to the robot; the analysis works that out from
                        // a standing reference.
  uint8_t  st;          // bit0-2 robot_state, bit3 torso_loop, bit4 probe_on.
                        // pengu-6 (2026-09-02) came back 5.7 s long when the bout was meant
                        // to be 25, and nothing in the file could say whether it had been
                        // stopped, re-READYed or never started -- so the mode now travels
                        // with every row.
};
Rec      rec[REC_MAX];
int      rec_head = 0, rec_count = 0;

// Parameter changes are sparse, so they live in their own small table instead of costing
// bytes in every sample. rec_dump interleaves them so each row carries what was in force.
struct Ev { uint32_t t_ms; float freq, leg, hip, phi, off; };
const int EV_MAX = 40;
Ev  ev[EV_MAX];
int ev_count = 0;

void rec_reset() { rec_head = 0; rec_count = 0; ev_count = 0; }

void rec_event(uint32_t t_ms) {
  if (ev_count >= EV_MAX) return;     // the table is a log of what was tried, not a ring:
  ev[ev_count].t_ms = t_ms;           // losing the FIRST settings would be worse
  ev[ev_count].freq = p_legFreq; ev[ev_count].leg = p_legAmp;
  ev[ev_count].hip  = p_hipAmp;  ev[ev_count].phi = p_hipPhi;
  ev[ev_count].off  = p_hipOff;
  ev_count++;
}

static inline int16_t q100(float v) { return (int16_t)constrain(v * 100.0f, -32000.0f, 32000.0f); }

void rec_push(float t, float phase, float slL, float slR, float hpL, float hpR,
              float torso, float gtorso, float mA) {
  Rec &r = rec[rec_head];
  r.t_ms   = (uint16_t)(t * 1000.0f);
  float ph = fmod(phase, 2.0f * PI); if (ph < 0) ph += 2.0f * PI;
  r.phase_q = (uint16_t)(ph * (65535.0f / (2.0f * PI)));
  r.slL = q100(slL); r.slR = q100(slR); r.hpL = q100(hpL); r.hpR = q100(hpR);
  r.torso = q100(torso); r.gtorso = q100(gtorso);
  r.roll = q100(imu_roll); r.pitch = q100(imu_pitch);
  r.mA20 = (int8_t)constrain(mA / 20.0f, -127.0f, 127.0f);
  r.gx = (int16_t)constrain(grav_x * 1000.0f, -32000.0f, 32000.0f);
  r.gy = (int16_t)constrain(grav_y * 1000.0f, -32000.0f, 32000.0f);
  r.gz = (int16_t)constrain(grav_z * 1000.0f, -32000.0f, 32000.0f);
  r.st   = (uint8_t)robot_state | ((torso_mode & 0x03) << 3) | (probe_on ? 0x20 : 0);
  rec_head = (rec_head + 1) % REC_MAX;
  if (rec_count < REC_MAX) rec_count++;
}

float rec_seconds() { return rec_count * TEL_MS / 1000.0f; }

// Rebuild the CSV. Slow on purpose: the robot is stopped, so there is no deadline, and a
// readable text file that phase_probe.py already parses is worth more than a compact one.
void rec_dump(WiFiClient client) {
  // A leading comment line, so a recording carries the settings that were flashed into
  // it. phase_probe.py already skips lines that are not CSV.
  {
    char h[192], *p = h;
    p += sprintf(p, "# kappa "); p = fmt(p, KAPPA, 2);
    p += sprintf(p, " kp ");     p = fmt(p, KP, 2);
    p += sprintf(p, " ki ");     p = fmt(p, KI, 2);
    p += sprintf(p, " kd ");     p = fmt(p, KD, 3);
    p += sprintf(p, " clamp ");  p = fmt(p, TORSO_CLAMP_DEG, 0);
    p += sprintf(p, " ffLP ");   p = fmt(p, FF_LP_HZ, 1);
    p += sprintf(p, " telms %d legP %d torsoP %d", (int)TEL_MS, (int)LEG_PGAIN, (int)TORSO_PGAIN);
    p += sprintf(p, " ffA ");    p = fmt(p, ff_amp, 1);
    p += sprintf(p, " ffphi ");  p = fmt(p, ff_phi, 0);
    p += sprintf(p, " ffgain "); p = fmt(p, ff_gain, 2);
    p += sprintf(p, "\n");
    client.print(h);
  }
  client.print(F("w,t,alpha,goal_slL,pos_slL,goal_slR,pos_slR,goal_hipL,pos_hipL,"
                 "goal_hipR,pos_hipR,goal_torso,pos_torso,mA_torso,imu_roll,imu_pitch,"
                 "axis,dt_ms,freq,leg_amp,hip_amp,hip_phi,hip_off,state,tloop,tmode,probe,"
                 "gx,gy,gz\n"));
  int start = (rec_count == REC_MAX) ? rec_head : 0;     // oldest surviving sample
  int ei = 0;
  float f = ev_count ? ev[0].freq : p_legFreq, A = ev_count ? ev[0].leg : p_legAmp;
  float H = ev_count ? ev[0].hip : p_hipAmp,  P = ev_count ? ev[0].phi : p_hipPhi;
  float O = ev_count ? ev[0].off : p_hipOff;
  uint16_t prev_t = 0;
  char b[256];
  for (int i = 0; i < rec_count; i++) {
    Rec &r = rec[(start + i) % REC_MAX];
    while (ei < ev_count && ev[ei].t_ms <= r.t_ms) {     // parameters in force at this row
      f = ev[ei].freq; A = ev[ei].leg; H = ev[ei].hip; P = ev[ei].phi; O = ev[ei].off; ei++;
    }
    float t     = r.t_ms / 1000.0f;
    float phase = r.phase_q * (2.0f * PI / 65535.0f);
    float alpha = constrain((t - T_RAMP - T_SETTLE) / T_BLEND, 0.0f, 1.0f);
    float off   = HIP_REST_DEG + (O - HIP_REST_DEG) * constrain(t / T_RAMP, 0.0f, 1.0f);
    float phi   = P * PI / 180.0f;
    // the same expressions run_walk() uses, so the commanded columns cannot drift from it
    float magL = alpha * 0.5f * A * (1.0f + sinf(phase));
    float magR = alpha * 0.5f * A * (1.0f + sinf(phase + PI));
    float hL   = off + alpha * H * max(0.0f, sinf(phase + PI + phi));
    float hR   = off + alpha * H * max(0.0f, sinf(phase + phi));
    float roll = r.roll / 100.0f, J = r.torso / 100.0f;
    char *p = b;
    p += sprintf(p, "w,");
    p = fmtc(p, t, 3);              p = fmtc(p, alpha, 3);
    p = fmtc(p, -magL, 2);          p = fmtc(p, r.slL / 100.0f, 2);
    p = fmtc(p, magR, 2);           p = fmtc(p, r.slR / 100.0f, 2);
    p = fmtc(p, -hL, 2);            p = fmtc(p, r.hpL / 100.0f, 2);
    p = fmtc(p, hR, 2);             p = fmtc(p, r.hpR / 100.0f, 2);
    p = fmtc(p, r.gtorso / 100.0f, 2); p = fmtc(p, J, 2);
    p = fmtc(p, r.mA20 * 20.0f, 0); p = fmtc(p, roll, 2);
    p = fmtc(p, r.pitch / 100.0f, 2);  p = fmtc(p, roll - J, 2);
    p = fmtc(p, (float)(uint16_t)(r.t_ms - prev_t), 1);
    p = fmtc(p, f, 3); p = fmtc(p, A, 1); p = fmtc(p, H, 1);
    p = fmtc(p, P, 0); p = fmtc(p, O, 0);
    int tm = (r.st >> 3) & 0x03;
    // tloop is kept as it was -- 1 when the kappa PID is the thing driving the torso --
    // so every analysis written before the feedforward mode existed still reads correctly
    p += sprintf(p, "%d,%d,%d,%d,", r.st & 0x07, tm == 0 ? 1 : 0, tm, (r.st >> 5) & 1);
    p = fmtc(p, r.gx / 1000.0f, 3); p = fmtc(p, r.gy / 1000.0f, 3);
    p = fmt(p, r.gz / 1000.0f, 3);
    *p++ = '\n'; *p = 0;
    prev_t = r.t_ms;
    client.print(b);
  }
}

// ------------------------------------------------------------------ IMU
// Only what the control law and the log use: Euler roll (the controlled quantity), pitch
// (lean readout), and the roll rate for the D term.
void update_imu() {
  imu::Vector<3> e = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
  imu_roll = e.y(); imu_pitch = e.z();          // same mapping as pengu.ino
  imu::Vector<3> g = bno.getVector(Adafruit_BNO055::VECTOR_GRAVITY);
  grav_x = g.x(); grav_y = g.y(); grav_z = g.z();

  // roll rate by differencing, lightly filtered. tau 30 ms is far below the gait period, so
  // it costs no phase where it matters, and it keeps the 0.07 deg quantisation steps from
  // showing up as spikes. Differencing rather than the gyro because its SIGN is right by
  // construction; gz is the matching gyro axis but only at slope -0.60 (measured), since
  // Euler rates are not body rates at this pitch.
  static unsigned long r_prev_ms = 0;
  static float roll_prev = 0.0f;
  unsigned long r_now = millis();
  if (r_prev_ms) {
    float rdt = (r_now - r_prev_ms) * 0.001f;
    if (rdt > 0.001f && rdt < 0.2f) {
      float w = rdt / (0.03f + rdt);
      roll_rate = (1.0f - w) * roll_rate + w * ((imu_roll - roll_prev) / rdt);
    }
  }
  r_prev_ms = r_now; roll_prev = imu_roll;
}


// ------------------------------------------------------------------ the gait
void run_walk() {
  float t = (millis() - walk_start_ms) / 1000.0f;

  // staged start: ramp the lean in -> settle -> blend the oscillation in
  float off_deg = HIP_REST_DEG + (p_hipOff - HIP_REST_DEG) * constrain(t / T_RAMP, 0.0f, 1.0f);
  float alpha   = constrain((t - T_RAMP - T_SETTLE) / T_BLEND, 0.0f, 1.0f);
  float phase   = (t > T_RAMP + T_SETTLE)
                  ? 2.0f * PI * p_legFreq * (t - T_RAMP - T_SETTLE) + gait_phi_off : 0.0f;

  // legs: unipolar antiphase   hips: half-rectified antiphase, offset by hip_phi
  float la = probe_on ? 0.0f : p_legAmp;      // a probe drives the torso and nothing else
  float ha = probe_on ? 0.0f : p_hipAmp;
  float magL = alpha * 0.5f * la * (1.0f + sinf(phase));
  float magR = alpha * 0.5f * la * (1.0f + sinf(phase + PI));
  float phi  = p_hipPhi * PI / 180.0f;
  float hipL_deg = off_deg + alpha * ha * max(0.0f, sinf(phase + PI + phi));
  float hipR_deg = off_deg + alpha * ha * max(0.0f, sinf(phase + phi));

  // torso: target world roll = KAPPA * hip-axis roll. The axis roll is reconstructed as
  // (torso roll - torso joint), which the 2026-08-29 mocap confirmed against directly
  // measured thigh attitude.
  float J_deg = dxl.getPresentPosition(XM_TORSO_ROLL, UNIT_DEGREE) - home_deg[idxOf(XM_TORSO_ROLL)];
  float axis  = imu_roll - J_deg;
  static unsigned long ctrl_prev_ms = 0;
  unsigned long ctrl_now_ms = millis();
  float ctrl_dt = ctrl_prev_ms ? (ctrl_now_ms - ctrl_prev_ms) * 0.001f : 0.02f;
  ctrl_prev_ms = ctrl_now_ms;
  if (ctrl_dt < 0.001f || ctrl_dt > 0.5f) ctrl_dt = 0.02f;    // ignore pauses / first call
  loop_dt_ema = (loop_dt_ema <= 0.0f) ? ctrl_dt : (0.9f * loop_dt_ema + 0.1f * ctrl_dt);

  // The feedforward's estimate of the hip-axis roll, low-passed. Everything else -- the
  // error, and so the whole KP path -- still sees the raw measurement.
  if (FF_LP_HZ > 0.0f) {
    float tau = 1.0f / (2.0f * PI * FF_LP_HZ);
    axis_lp += (ctrl_dt / (ctrl_dt + tau)) * (axis - axis_lp);
  } else {
    axis_lp = axis;
  }

  float torso_deg = 0.0f;                    // loop open: hold home, run no controller
  if (probe_on) {
    float tp = t - T_RAMP - T_SETTLE;
    int k = 0;
    float acc = 0.0f;
    while (k < PROBE_N && tp > acc + PROBE_SEC[k]) { acc += PROBE_SEC[k]; k++; }
    if (k >= PROBE_N) {                      // sweep finished: stop, so the ring keeps it
      robot_state = STATE_IDLE;
      tx("# probe complete -> IDLE. Press DOWNLOAD.");
      return;
    }
    probe_phase += 2.0f * PI * PROBE_HZ[k] * ctrl_dt;
    torso_deg = alpha * PROBE_AMP_DEG * sinf(probe_phase);
  } else if (torso_mode == TM_FF) {
    // locked to the SAME phase variable the legs use, so a bumpless frequency change
    // carries the torso with it
    torso_deg = alpha * ff_gain * ff_amp * sinf(phase + ff_phi * PI / 180.0f);
    torso_deg = constrain(torso_deg, -TORSO_CLAMP_DEG, TORSO_CLAMP_DEG);
    torso_iErr = 0.0f;
  } else if (torso_mode == TM_PID) {
    float err  = KAPPA * axis_lp - imu_roll;
    torso_iErr = constrain(torso_iErr + err * ctrl_dt, -20.0f, 20.0f);
    torso_deg  = alpha * constrain(
        (KAPPA - 1.0f) * axis_lp + KP * err + KI * torso_iErr + KD * (-roll_rate),
        -TORSO_CLAMP_DEG, TORSO_CLAMP_DEG);
  } else {
    torso_iErr = 0.0f;                       // no windup to dump when it is closed again
  }

  // sign conventions inherited from pengu.ino: left negative, right positive
  dxl.setGoalPosition(XM_LEFT_SLIDE,  home_deg[idxOf(XM_LEFT_SLIDE)]  - magL,      UNIT_DEGREE);
  dxl.setGoalPosition(XM_RIGHT_SLIDE, home_deg[idxOf(XM_RIGHT_SLIDE)] + magR,      UNIT_DEGREE);
  dxl.setGoalPosition(XM_LEFT_HIP,    home_deg[idxOf(XM_LEFT_HIP)]    - hipL_deg,  UNIT_DEGREE);
  dxl.setGoalPosition(XM_RIGHT_HIP,   home_deg[idxOf(XM_RIGHT_HIP)]   + hipR_deg,  UNIT_DEGREE);
  dxl.setGoalPosition(XM_TORSO_ROLL,  home_deg[idxOf(XM_TORSO_ROLL)]  + torso_deg, UNIT_DEGREE);

  // ---- record into RAM. No radio traffic here: that is the whole point of buffering.
  // Recording starts a second before the blend, not at t=0. The staged start is 14 s and
  // the ring holds about 15.8, so a bout logged from zero comes back almost entirely
  // ramp-and-settle: pengu-8 had 53 of 560 rows at alpha=1. Skipping the quiet part gives
  // a full ring of the gait itself and removes the need to time the STOP by hand.
  static unsigned long tel = 0;
  if (t >= T_RAMP + T_SETTLE - 1.0f && millis() - tel >= TEL_MS) {
    tel = millis();
    rec_push(t, phase,
             dxl.getPresentPosition(XM_LEFT_SLIDE,  UNIT_DEGREE) - home_deg[idxOf(XM_LEFT_SLIDE)],
             dxl.getPresentPosition(XM_RIGHT_SLIDE, UNIT_DEGREE) - home_deg[idxOf(XM_RIGHT_SLIDE)],
             dxl.getPresentPosition(XM_LEFT_HIP,    UNIT_DEGREE) - home_deg[idxOf(XM_LEFT_HIP)],
             dxl.getPresentPosition(XM_RIGHT_HIP,   UNIT_DEGREE) - home_deg[idxOf(XM_RIGHT_HIP)],
             J_deg, torso_deg,
             (int16_t)dxl.readControlTableItem(PRESENT_CURRENT, XM_TORSO_ROLL) * 2.69f);
  }
}


// Commands are applied HERE, not in loop(), so the HTTP reply carries the state the press
// produced rather than the one before it. wireless.ino calls it from the /cmd handler.
void apply_cmd(char cmd) {
  switch (cmd) {
    case 'r': robot_state = STATE_READY_SLIDE; tx("-> READY"); break;
    case 'w':
      if (robot_state == STATE_IDLE) {
        walk_start_ms = millis(); torso_iErr = 0; gait_phi_off = 0.0f; axis_lp = 0.0f;
        rec_reset(); rec_event(0);          // event 0 = the settings the bout started on
        robot_state = STATE_WALK;
        tx(torso_mode == TM_FF && KAPPA != 0.0f
           ? "-> WALK  ** feedforward is calibrated for KAPPA=0 only **"
           : "-> WALK (staged start: 4 s lean, 6 s settle, 4 s blend)");
        announce();
      } else tx("Go READY first.");
      break;
    case 'q': robot_state = STATE_IDLE; tx("-> IDLE"); break;

    // leg_amp is allowed to go negative. The crank command is 0.5*A*(1+sin), so the sign
    // decides which way the crank turns away from its home angle. In the model that makes
    // no difference at all -- home sits on the crank-slider's dead centre, so turning
    // either way pulls the slider the same way and +75 and -75 produce the same stroke
    // (12.8 vs 13.1 mm) and the same gait (v_net 0.1041 vs 0.1041 at 1.46/250/./32/20).
    // If it DOES make a difference on the robot, the home angle is not on the dead centre
    // and the executed leg motion is not the commanded waveform.
    case 'j': p_legAmp = max(-180.0f, p_legAmp - STEP_LEG); announce(); break;
    case 'k': p_legAmp = min(180.0f, p_legAmp + STEP_LEG); announce(); break;
    case 'n': p_hipAmp = max(0.0f, p_hipAmp - STEP_HIP);   announce(); break;
    case 'm': p_hipAmp = min(45.0f, p_hipAmp + STEP_HIP);  announce(); break;
    case ',': p_hipPhi = fmod(p_hipPhi - STEP_PHI + 360.0f, 360.0f); announce(); break;
    case '.': p_hipPhi = fmod(p_hipPhi + STEP_PHI, 360.0f);          announce(); break;
    case '[': setFreq(p_legFreq - STEP_FREQ); announce(); break;
    case ']': setFreq(p_legFreq + STEP_FREQ); announce(); break;
    case 'o': p_hipOff = max(0.0f,  p_hipOff - STEP_OFF); announce(); break;
    case 'O': p_hipOff = min(70.0f, p_hipOff + STEP_OFF); announce(); break;
    case '0': p_legAmp = 0.0f; p_hipAmp = 0.0f;
              tx("# amplitudes zeroed (still walking, still recording)"); announce(); break;
    // The two GRID-6 champions at mu = 0.5 on models/hardware_c1 (2026-09-02).
    // Both were re-measured for the things the sweep's pass gate does not check --
    // whether the feet leave the ground every cycle, and whether the body's roll stays
    // locked to the gait -- because the sweep's first mu=0.5 winner failed both: a step
    // where the foot never rose above its loaded height, and a roll phase wandering
    // 25.5 deg per cycle. These two do not.
    // Picked from the first sweep run under the torso this robot actually has -- 108,284
    // cells, all of them carrying the 56 ms torso lag, the leg servos modelled rather than
    // fenced off (354 deg/s slew plus a 29 ms pole, calibrated against two hardware points
    // on opposite sides of the ceiling), and the feedforward torso rather than the kappa
    // PID. Ranked on net speed with foot clearance, roll phase drift and torso roll all
    // measured alongside.
    //
    // Both of these ASK FOR MORE CRANK THAN THE SERVO CAN GIVE, deliberately: 572 and 588
    // deg/s against the measured 354. The status line will say OVER. Fencing those cells
    // off is what held the previous answer to 0.128 m/s; simulating the clipping instead
    // is where these came from. What the robot executes will be a flattened, delayed
    // version of the commanded 130 -- roughly 0.55 of it, arriving about 90 ms late.
    //
    // Each carries its own feedforward. ff_phi is a STARTING POINT from the model, not a
    // measurement: the axis roll of these gaits is only 22-28% sinusoidal (against 95% for
    // the calibration gait on preset 3), so a single-frequency feedforward cancels less of
    // it and the phase will want scanning with h/H before the number means anything.
    case '1':   // c3 cap-only champion, mu 0.12 ice: 1.25 / 280 / 125 / 24 / 20, sim 0.142 m/s, 13 mm worst-cycle clearance,
                // torso roll 2.5 deg rms, nbhd pass 0.89, crank demand 491 deg/s (cap 354, modelled).
      setFreq(1.25f); p_legAmp = 125.0f; p_hipAmp = 24.0f; p_hipPhi = 280.0f; p_hipOff = 20.0f;
      torso_mode = TM_PID;
      announce(); break;
    case '2':   // c3 cap-only champion, mu 0.45 floor: 1.65 / 310 / 130 / 20 / 20, sim 0.248 m/s, 39 mm worst-cycle clearance,
                // torso roll 5.8 deg rms, nbhd pass 0.89, crank demand 674 deg/s (cap 354, modelled).
      setFreq(1.65f); p_legAmp = 130.0f; p_hipAmp = 20.0f; p_hipPhi = 310.0f; p_hipOff = 20.0f;
      torso_mode = TM_PID;
      announce(); break;
    case '3':   // the torso calibration gait, not a candidate: exactly what pengu-12 ran
                // with the loop held, the only bout so far that never fell (max total tilt
                // 11.1 deg over 15 s) and the one whose axis roll is clean enough to fit
                // -- 5% residual, against 72-78% for presets 1 and 2. Come back here
                // whenever a feedforward phase needs re-checking against something known.
      setFreq(1.39f); p_legAmp = 80.0f; p_hipAmp = 16.0f; p_hipPhi = 240.0f; p_hipOff = 30.0f;
      ff_amp = 7.5f; ff_phi = 119.0f; ff_gain = 1.0f;
      announce(); break;
    case 'F': p_legAmp = -p_legAmp; announce(); break;
    case 'T': torso_mode = (torso_mode + 1) % 3;
              torso_iErr = 0.0f; axis_lp = 0.0f; announce(); break;
    case 'Q': ff_phi = fmod(ff_phi + 45.0f, 360.0f);         announce(); break;
    case 'h': ff_phi = fmod(ff_phi - 5.0f + 360.0f, 360.0f); announce(); break;
    case 'H': ff_phi = fmod(ff_phi + 5.0f, 360.0f);          announce(); break;
    case 'a': ff_amp = max(0.0f,  ff_amp - 0.5f);            announce(); break;
    case 'A': ff_amp = min(25.0f, ff_amp + 0.5f);            announce(); break;
    case 'g': ff_gain = max(0.0f, ff_gain - 0.05f);          announce(); break;
    case 'G': ff_gain = min(2.0f, ff_gain + 0.05f);          announce(); break;
    case 'P': probe_on = !probe_on; probe_phase = 0.0f; torso_iErr = 0.0f; axis_lp = 0.0f;
              announce(); break;
    case 'i': robot_state = STATE_IDLE; tx("# re-initialising motors"); initMotors(); break;
  }
}


// ------------------------------------------------------------------ main loop
void loop() {
  update_imu();

  update_wifi();

  switch (robot_state) {
    case STATE_IDLE: break;
    case STATE_READY_SLIDE:
      stepMotorToward(XM_LEFT_SLIDE,  0.0f, READY_STEP);
      stepMotorToward(XM_RIGHT_SLIDE, 0.0f, READY_STEP);
      if (arrivedAt(XM_LEFT_SLIDE, 0.0f) && arrivedAt(XM_RIGHT_SLIDE, 0.0f)) {
        robot_state = STATE_READY_HIP; tx("Slides at zero -> hips (rest lean)");
      }
      break;
    case STATE_READY_HIP:
      stepMotorToward(XM_LEFT_HIP, 360.0f - HIP_REST_DEG, READY_STEP);   // L: -10 deg (fwd)
      stepMotorToward(XM_RIGHT_HIP, HIP_REST_DEG,         READY_STEP);   // R: +10 deg (fwd)
      if (arrivedAt(XM_LEFT_HIP, 360.0f - HIP_REST_DEG) && arrivedAt(XM_RIGHT_HIP, HIP_REST_DEG)) {
        robot_state = STATE_READY_TORSO; tx("Hips at rest lean -> torso");
      }
      break;
    case STATE_READY_TORSO:
      stepMotorToward(XM_TORSO_ROLL, 0.0f, READY_STEP_TORSO);
      if (arrivedAt(XM_TORSO_ROLL, 0.0f)) { robot_state = STATE_IDLE; tx("READY done -> IDLE"); }
      break;
    case STATE_WALK: run_walk(); break;
  }

  set_status_rgb(motors_ok ? (robot_state == STATE_WALK ? 2 : 1) : 0);

  // No pacing delay: the gait phase comes from absolute time, so a delay would change the
  // control rate without changing the waveform.
  loop_count++;
  static unsigned long hz_t = 0;
  if (millis() - hz_t >= 1000) { hz_t = millis(); loop_hz = loop_count; loop_count = 0; }
}
