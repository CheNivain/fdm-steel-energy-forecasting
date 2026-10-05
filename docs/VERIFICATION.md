# Application Verification - 5 October 2026

## Artifacts

The original evaluation artifact is unchanged:
861284d63c80b31af840333e893d92f7f3906ebe86633edfa2213e3f8d1257a2

The new full-year deployment artifact:
396c0fb5ed61da744a8fd88513615643d5e170fd4db72b981b468d077698da5a

The active deployment manifest is app_data/deployment/manifest.json.
Its fixed-setting 1,000-tree fit used 33,695 examples from the 35,040 original
readings and took 503.14 seconds. The first 672 rows and 673 rows affected by
the unverified reading were excluded. No future-value interpolation was used
for this refit. It has no claimed independent future accuracy score.

## Python tests

All 51 tests passed with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Coverage includes existing model-development tests plus causal feature agreement,
real saved-model replay, CSV validation, exact first future interval, recursive
lag reuse, 18:00 jumps, 30-minute requests, immutable reconciliation, correction
audits, persisted state, duplicate request handling, stale-client rejection,
concurrent requests, atomic prediction failure and actual-only background refits.

The background refit state-machine test uses a controlled test double; the full
1,000-tree original-data deployment fit was executed separately and its loaded
artifact was exercised by browser tests. No claim is made of having real 2019 data.

## Browser acceptance

Both browser suites passed:

- tests/browser_rolling.cjs: future workflow using the real deployment artifact.
- tests/browser_smoke.cjs: existing replay/import/export workflow.

The rolling suite verified all seven weekday examples, future predictions with
unknown actuals, recursive 30-minute progression, invalid CSV rejection, actual
reconciliation, cleared stale results, first-issued error reporting, feature
inspection, CSV export, reload persistence and the five-model comparison table.
Plant-clock arithmetic was checked in a browser using America/New_York across a
daylight-saving transition. The forecast chart contained 106,422 painted pixels.

All five pages fit 320, 390, 768, 1024 and 1920 pixel widths without document
overflow. A 1440-pixel desktop was also exercised. Screenshots were reviewed.
Both suites reported no unexpected JavaScript errors.

Evidence:
- tmp/final-system-qa/browser-results.json and PNG screenshots.
- tmp/app-qa/browser-results.json and PNG screenshots for legacy coverage.

Tests used isolated databases. The synthetic QA actual (55 kWh) was never added
to the user's main workspace. At delivery verification, that main workspace had
35,040 original readings, one flagged unverified reading, zero rolling forecasts,
zero actual imports, and a first future interval of 2019-01-01 00:00-00:15.

## Running app

The finished local app is served at http://127.0.0.1:8002.
Both model integrity/version checks passed, and deployment_ready is true.
Historical evaluation metrics remain labelled as development results, not
accuracy evidence for the full-year refit or recursive forecasts.

This verification demonstrates local functionality. It does not establish
reliable month-long prediction chains, production certification, a common
five-model ranking, guaranteed savings or a particular assessment mark.
