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

By default, the simulator places `NODE_04`, `NODE_05`, and `NODE_06` in a
140-metre GPS cluster inside FUT Minna's Gidan Kwano main campus. Override
`FUT_MINNA_CENTER_LATITUDE`, `FUT_MINNA_CENTER_LONGITUDE`, or
`FUT_MINNA_NODE_RADIUS_METERS` when a different on-campus plot is required.
