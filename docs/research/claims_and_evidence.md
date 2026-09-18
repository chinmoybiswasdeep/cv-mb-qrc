# Claims and evidence matrix

| Claim | Status | Required evidence |
|---|---|---|
| Gaussian CV and affine twin agree within strict floating-point tolerance | Supported by validation tests; publication figure requires final run | state errors at every step, Tier A/B, size/noise/loss sweep, paired predictions |
| Gaussian task performance implies quantum advantage | Unsupported | Gaussian dynamics have an exact efficient classical twin |
| Capacity exceeds calibrated null behavior | Undetermined until confirmatory run | raw R², all null scores/descriptors, empirical p, BH q, calibrated default null |
| One method outperforms another | Undetermined | paired hierarchical interval, effect size, cluster test and corrected q-value |
| Physical tap-homodyne readout is implemented | Unsupported | backend instrument and backaction validation are absent |
| Graphix backend is generic MBQC | Unsupported | execution is scoped to a one-memory collision family; guarded topologies are design validation only |
| Graphix and MentPy agree generally | Unsupported | common evidence is limited to corrected pure XY wires |
| Fock results show non-Gaussian advantage | Unsupported | current stage is cutoff/postselection feasibility only |
| Evidence bundle is internally intact | Supported only after `verify-run` succeeds | content index, reconstructed tables, source and figure hashes |

Generated runs contain the machine-readable counterpart at
`json/claims_evidence.json`, figure captions at `json/figure_captions.json`, and
panel-to-table mappings at `json/figure_provenance.json`.
