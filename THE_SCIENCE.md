## Background

**Targets and pockets.** A protein folds into a 3D shape with concave pockets
on its surface. For an enzyme, one of them is the active site. EV-A71 2A
protease cuts the virus's own polyprotein into working parts. Block that site
and the virus cannot assemble.

**Pose and affinity.** A bound molecule is held by hydrogen bonds, packed
hydrophobic surfaces, sometimes a metal ion. Where and how it sits is the pose.
How tightly it is held is the affinity. The 2 are measured separately, and a
molecule can get one right and the other wrong.

**Crystallographic ground truth.** Grow a crystal, soak the compound in,
reconstruct where every atom sits. Slow and expensive, which is why 925 binding
events on 1 target is a notable public dataset.

**Co-folding prediction.**

![How a co-folding model predicts a binding pose](docs/images/cofolding_explained.gif)

A co-folding model gets the protein sequence and the ligand SMILES. No structure
for either. It solves both at once, so the pocket can open to accept the ligand
instead of staying fixed. That matters here: a loop beside the 2A active site
sits closed in the empty protein and swings open on binding, and docking into a
fixed receptor fails on exactly that.

**Sequence alignments.** The model also gets a multiple sequence alignment,
hundreds of versions of the same protein from other species. Positions that
mutate together are in contact in 3D. That signal is where most of the folding
accuracy comes from. We precompute 1 alignment per target, because every complex
shares the same protein.

**Pose scoring.** Align the predicted protein onto the crystal protein, then
measure the average distance between matched ligand atoms. Under 2 A is the
field's threshold for correct, about the width of a chemical bond. Above it, the
molecule is in the pocket but turned the wrong way, and anything a chemist
concludes from it is wrong too.

**Discovery campaigns.**

![From fragment screen to optimized compound](docs/images/campaign_explained.gif)

Discovery starts with a fragment screen: molecules of about 15 heavy atoms,
soaked into crystals by the thousand. Most bind nothing. The few that do bind
weakly, and they show which sub-pocket accepts chemistry. Chemists grow those
fragments outward round after round, gaining potency each time.

**Generalization.** These models were trained on the PDB, dense in some protein
families and sparse elsewhere. A model can score well on a benchmark that
resembles its training data and fail on a target it has not effectively
memorized. The OpenBind complexes sit at a median similarity of about 40 to
pre-2021 public structures, on a scale where co-folding success runs 80 to 90
percent above 80 and 10 to 40 percent below 50. Splitting by campaign stage
measures the same property from the other direction.

**Affinity prediction.** Pose is geometry. Affinity is free energy, including
entropy, water displaced from the pocket, and the protein's own flexibility. The
most useful result in the OpenBind paper is a negative one: a molecular weight
baseline beat every structure-based method on rank correlation. Bigger molecules
tend to bind harder, and a model that learns only that has learned nothing about
the pocket.

**Target-specific fine-tuning.** 79 fragment structures teach the model what
this pocket looks like when it is open and what chemistry it accepts. That
transfers to the 802 follow-ons, which share the pocket even when they do not
share the chemistry. Small data doing a lot of work, and the result most
relevant to a discovery team, because every campaign generates fragment
structures early.
