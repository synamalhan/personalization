# On-device measurement protocol (experiment 7)

Write-once rules. Follow them every time, and record any deviation next to the results.

## Setup

- **Device:** a real iPhone, not the Simulator. Record the model identifier (the benchmark CSV
  includes it, for example `iPhone16,1`) and the iOS version.
- **Build:** Release configuration (Product > Scheme > Edit Scheme > Run > Build Configuration = Release).
  Debug builds are 5-50x slower in plain Swift loops and must not be reported. The CSV records
  `build` and a report must say `release`.
- **Conditions:** airplane mode optional (the app uses no network), Low Power Mode off, screen on
  and unlocked, app in the foreground, charger unplugged, other apps closed, device at room temperature.
- **Thermal state:** record `ProcessInfo.thermalState` before and after the run (columns
  `thermal_before`, `thermal_after`). Start only from `nominal`. If it is not `nominal` before the run,
  let the phone cool and repeat. Report any run that ended above `nominal`.

## Procedure

1. Open Settings > Lab > Benchmark. Tap **Run benchmark** once per session.
2. For every metric the app runs **10 warm-up iterations, discarded**, then **100 timed iterations**.
   (This is also why the Core ML model is loaded before timing: the first call compiles/loads and is
   discarded as warm-up.)
3. Metrics:
   - `embed_<backbone>`: preprocessing (resize 256, center crop 224) plus Core ML prediction for one
     bundled photo, per installed backbone.
   - `teach_5_<head>` and `teach_20_<head>`: `Learner.teach` (add vectors and refit) for 5 and 20
     photos on top of 10 existing classes with 5 examples each (960-d), for both heads. Embedding time is
     excluded; it is measured by `embed_*`.
4. Memory: the app's physical footprint (`phys_footprint`) before and after the run.
5. Repeat the whole run **3 times in separate app launches**, resting the phone between runs, and report
   all three, not only the best.

## Reporting

- Report the **median and 95th percentile** (nearest-rank) of the 100 timed iterations per run, in
  milliseconds. Do not report the mean alone and do not drop outliers.
- State: device model, iOS version, build configuration, thermal state before and after, number of
  warm-up and timed iterations, and the model precision (fp16 or int8) of the bundled backbone.
- Raw timings stay in the exported CSV (`benchmark_release.csv`); copy them to `results/` unmodified.

## Caveats to state with the numbers

- One device, one OS version.
- The timed photo is a fixed bundled image; real photos may take slightly different preprocessing time.
- Core ML may schedule work on the Neural Engine, GPU or CPU; the compute unit is chosen by the system
  (`.all`) and is not recorded.
- Timings come from `ContinuousClock` around the call (wall time), not from Instruments.

## How to run in Release (Xcode)

1. Product > Scheme > Edit Scheme... > Run > Info > Build Configuration: **Release**.
2. Select your iPhone as the run destination, then Product > Run (Cmd+R). First run on a device needs a
   signing team (Signing & Capabilities) and Developer Mode (Settings > Privacy & Security).
3. Go to Settings > Lab > Benchmark > Run benchmark, then use the share button to AirDrop the CSV.
4. Switch the scheme back to Debug afterwards.
