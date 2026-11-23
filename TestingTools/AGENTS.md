# TestingTools — agent notes

iOS app for live inference and grading. Layout is per-model (`feature_layout` / `preferredLayout`). v2 models use `LiveSegmenter` + `SequencePreprocessorV2`. Phase2ParityCheck must use `.cpuOnly` on simulator.

After `./lsm deploy`, register new bundles in `LandmarkPipeline.swift` if the snippet was not applied automatically. Xcode signing stays local — do not commit personal team IDs.
