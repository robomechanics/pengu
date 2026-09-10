// wireless.ino — the AP, the buttons, and the recording download.
//
// Nothing is transmitted while the robot walks. The run is captured into a RAM ring buffer
// (rec_push, main tab) and fetched afterwards over HTTP. An earlier version of this file
// streamed every sample live over UDP; that costs an SPI transaction to the NINA module
// inside the control loop, on the order of 5-15% of the loop period at 50 Hz -- a
// perturbation to the very quantity the recording exists to measure. Now the only traffic
// during a bout is the occasional button press.
//
// Two things here are load-bearing for the control loop, and both were bugs in the sketch
// this is descended from:
//
// 1. The old update_wifi() waited up to 1000 ms for a client's HTTP headers. One second of
//    blocking inside a 50-90 Hz control loop is not a hiccup, it is a fall. The read budget
//    here is HTTP_BUDGET_MS and an incomplete request is dropped; the browser retries.
// 2. The page is several kilobytes pushed through TCP in small chunks, which blocks for as
//    long as it takes. It is refused while walking -- the page is loaded once before the
//    run and does not need reloading during it.
//
// /dump is refused while walking for the same reason; it is a long transfer by design.

#include <WiFiNINA.h>
#include <utility/wifi_drv.h>

const int RGB_RED = 25, RGB_GREEN = 26, RGB_BLUE = 27;   // on the NINA module

const char     AP_SSID[]      = "PENGU";
const uint16_t HTTP_BUDGET_MS = 10;        // hard cap on header reading, see note above

WiFiServer server(80);
bool       wifi_active = false;

extern RobotState robot_state;             // to refuse long transfers while walking


void set_rgb(int r, int g, int b) {
  WiFiDrv::analogWrite(RGB_RED, r);
  WiFiDrv::analogWrite(RGB_GREEN, g);
  WiFiDrv::analogWrite(RGB_BLUE, b);
}

// 0 = motors not holding (red), 1 = ready (green), 2 = walking (blue).
// The only status indicator left now that there is no terminal.
void set_status_rgb(int state) {
  static int last = -1;
  if (state == last) return;
  last = state;
  if (state == 0)      set_rgb(24, 0, 0);
  else if (state == 1) set_rgb(0, 16, 0);
  else                 set_rgb(0, 0, 24);
}

void begin_wifi() {
  WiFiDrv::pinMode(RGB_RED, OUTPUT);
  WiFiDrv::pinMode(RGB_GREEN, OUTPUT);
  WiFiDrv::pinMode(RGB_BLUE, OUTPUT);
  set_rgb(24, 0, 0);

  if (WiFi.status() == WL_NO_MODULE) return;
  WiFi.config(IPAddress(192, 168, 4, 1));
  if (WiFi.beginAP(AP_SSID) != WL_AP_LISTENING) return;
  server.begin();
  wifi_active = true;
}

// Applies any command that arrived and replies with the resulting state. Never blocks for
// more than HTTP_BUDGET_MS except when serving a page or a dump, neither of which is
// served while walking.
void update_wifi() {
  if (!wifi_active) return;
  WiFiClient client = server.available();
  if (!client) return;

  char req[96];
  int n = 0;
  unsigned long t0 = millis();
  bool done = false;
  while (client.connected() && millis() - t0 < HTTP_BUDGET_MS) {
    while (client.available()) {
      char c = client.read();
      if (n < (int)sizeof(req) - 1) req[n++] = c;      // only the request line matters
      if (c == '\n') { done = true; break; }
    }
    if (done) break;
  }
  req[n] = 0;
  if (!done) {                       // ran out of budget: drop it, the browser will retry
    client.stop();
    return;
  }

  if (strstr(req, "GET /cmd")) {
    // The page sends the key through encodeURIComponent, which leaves letters, digits and
    // -_.!~*'() alone but escapes everything else. Reading k[4] raw therefore worked for
    // every alphanumeric button and silently returned '%' for the three that are not:
    // '[' and ']' (frequency) and ',' (hip_phi down) arrived as %5B, %5D, %2C. Decoding
    // here fixes the whole class rather than renaming the keys away from it.
    char *k = strstr(req, "key=");
    char cmd = k ? k[4] : 0;
    if (cmd == '%' && isxdigit((unsigned char)k[5]) && isxdigit((unsigned char)k[6])) {
      char hx[3] = { k[5], k[6], 0 };
      cmd = (char)strtol(hx, NULL, 16);
    }
    if (cmd) apply_cmd(cmd);          // act first, then report -- otherwise the page shows
                                      // the state from before the button was pressed
    char st[192];
    state_str(st);
    client.print(F("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n"));
    client.print(st);

  } else if (strstr(req, "GET /dump")) {
    if (robot_state == STATE_WALK) {
      client.print(F("HTTP/1.1 503 Busy\r\nContent-Type: text/plain\r\n"
                     "Connection: close\r\n\r\nstop the robot first"));
    } else {
      client.print(F("HTTP/1.1 200 OK\r\nContent-Type: text/csv\r\n"
                     "Content-Disposition: attachment; filename=pengu.csv\r\n"
                     "Connection: close\r\n\r\n"));
      rec_dump(client);               // slow by design; the robot is not walking
    }

  } else if (robot_state == STATE_WALK) {
    client.print(F("HTTP/1.1 503 Busy\r\nContent-Type: text/plain\r\n"
                   "Connection: close\r\n\r\nwalking"));
  } else {
    send_webpage(client);
  }
  client.stop();
}
