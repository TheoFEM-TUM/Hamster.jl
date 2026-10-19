# ====================
# ActiveLearning defaults
# ====================
@configtag nrounds Int64 5 "number of active learning rounds (each round: fit, then add structures)." "ActiveLearning"
@configtag nadd Int64 5 "number of structures added to the training set per round." "ActiveLearning"
@configtag nval Int64 20 "number of validation structures (fixed, never in the training set)." "ActiveLearning"
@configtag selection String "coverage" "selection of new structures (coverage=lowest mean coverage by the kernel points, random)." "ActiveLearning"
@configtag npoints_add Int64 0 "maximal number of kernel points added per round (farthest from the existing points)." "ActiveLearning"
@configtag add_below Float64 0.5 "kernel points are only added for descriptors with a similarity below this value." "ActiveLearning"
