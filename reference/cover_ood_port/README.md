# CoVer OOD env port (redbull/zucchini/tennis)

Source: github.com/cover-vla/cover-vla (vendored ManiSkill2_real2sim), extracted 2026-07-16.
- cover_ood_envs_excerpt.py: their env classes (PutRedbullOnPlateInScene,
  PutTennisBallIntoBasketInScene(?), PutZucchiniOnTowelInScene(?) — verbatim excerpt,
  lines 560-990 of their put_on_in_scene.py) incl. registrations.
- models/: the asset dirs their envs load.
- asset_info_entries.json: entries to merge into info_bridge_custom_v1.json.

Install on a sim pod (INT-ACT stack):
1. cp -r models/* /workspace/INT-ACT/ManiSkill2_real2sim/data/custom/models/  (path may differ)
2. merge asset_info_entries.json into data/custom/info_bridge_custom_v1.json
3. append the env classes (adapting imports/base classes to the INT-ACT fork's
   put_on_in_scene.py / put_on_in_new.py conventions) + register.
4. settle-test + one nominal rollout before use.
Nominal instructions (theirs): "put redbull can on plate", "put tennis ball into
yellow basket", "put the zucchini on the towel".
