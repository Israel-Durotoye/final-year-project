# Soil Doctor

This workspace contains the Soil Doctor frontend. I updated the branding
and switched the UI to a purple-first theme; favicon and OG image were
replaced for local development.

## Mixed hardware and simulator telemetry

The frontend combines the Firebase hardware feed and Supabase simulator feed
into the existing UI schema:

- `NODE_01` and `NODE_02` come from the Firebase Realtime Database log already
  written by the physical gateway firmware.
- `NODE_03` through `NODE_06` come from the simulator project's
  `capstone_dataset` table.

Copy `.env.example` to `.env` and configure the simulator Supabase project and
the physical gateway's Firebase Realtime Database URL.

Start the remaining simulated nodes with:

```bash
python sensor_simulator.py
```

The application consumes the INO firmware's existing
`/readings/log.json` Firebase output without requiring firmware changes. It
maps the snake_case hardware fields to the frontend telemetry schema and uses
the Firebase push ID as wall-clock time because the firmware timestamp is
device uptime.

The dashboard and node page refresh telemetry every 30 seconds; the map and
active-node counters also refresh automatically. Hardware rows are normalized
in `src/lib/telemetry.ts`, so the rest of the frontend can continue using the
existing `Node_ID`, `Timestamp`, and sensor column names.

Dashboard Field Reports are generated only when **Get Field Report** is pressed.
A successful report is retained per node in the browser for 30 minutes, so
telemetry refreshes do not continually replace it.

Map View displays values for all six sensor measurements and compares them with
the alert limits configured in Settings. Select a layer and a marker to see its
interpretation, suggested checks, and the nearest current sensors of the same
source type. Stale, missing, and invalid readings are excluded from numeric map
comparisons; nodes without valid coordinates remain selectable in the location
cards. Marker colours describe measured points, not interpolated field areas.

The homepage requests `field_summary` mode: one plain-language sentence, at most
45 words, covering the selected field area's main condition or change over the
available recorded period and one action when needed. It uses recent history
without inventing a time window; stale or missing readings take precedence over
treatment advice. Invalid or lengthy output is replaced with a concise summary
from sensor evidence, identified as `sensor-screening-report` in API metadata.
The separate `field_report` mode remains available for detailed assessments.

The primary text model is `deepseek-v4-flash` through AgentRouter, configurable
with `AGENTROUTER_MODEL`. On 2026-09-17, AgentRouter's
[live catalogue](https://agentrouter.org/api/pricing) included DeepSeek but no GLM
model. A synthetic four-section report and a tool-call follow-up both passed
live checks. These checks establish compatibility, not an uptime guarantee.
DeepSeek requests default to non-thinking mode so the 1,024-token output budget
is available for the visible answer; explicit caller settings can override it.
The existing Conduit and local fallbacks remain enabled when configured.
Restart the backend after changing `.env` model settings.

By default, the simulator places `NODE_04`, `NODE_05`, and `NODE_06` in a
140-metre GPS cluster inside FUT Minna's Gidan Kwano main campus. Override
`FUT_MINNA_CENTER_LATITUDE`, `FUT_MINNA_CENTER_LONGITUDE`, or
`FUT_MINNA_NODE_RADIUS_METERS` when a different on-campus plot is required.
