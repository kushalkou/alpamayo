AUDIT_SIGNOFF.md
Traces signed off: data/ar1_coc_2hz_v3.pkl
Audit 1 (v2 page, 100 traces): found 3 systematic rule bugs (route command in causes,
rear-axle distances / out-of-path yield candidates, turns outside intersection
polygons). Fixed in v3 (R11).
Audit 2 (v3 short page, 32 cards): 0 wrong decisions, 0 wrong causes, 8 unsure.
Pre-registered rule (wrong decisions <= 2 AND no cause error type >= 3): PASS.
Known limitations: nuScenes has no traffic-light state; whole-scene parked clips
(0771, 0777) and scene-0778 have ambiguous driver intent.
Signed: Kushal Koujalagi, 2026-10-12
