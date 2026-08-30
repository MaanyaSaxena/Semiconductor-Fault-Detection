# Data

The project uses the SECOM dataset from the UCI Machine Learning Repository.

The raw dataset is not included in this repository. The notebook downloads the data
from the UCI source when required.

Dataset characteristics used in the analysis:
- 1,567 production units
- 590 anonymized sensor measurements
- 104 failures (6.64%)
- 1,463 passing units (93.36%)

The sensor identities are anonymized, so model explanations identify sensor columns
rather than physical manufacturing components.

See the notebook for the complete data-loading, cleaning, modeling, and evaluation
pipeline.
