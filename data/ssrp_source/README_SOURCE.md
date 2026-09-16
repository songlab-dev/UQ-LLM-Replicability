# Source

The `D1`--`D8` files in the parent `data/` directory are the unmodified
aggregate-data bundle from the Social Sciences Replication Project (SSRP):

- Camerer et al. (2018), *Evaluating the replicability of social science
  experiments in Nature and Science between 2010 and 2015*, *Nature Human
  Behaviour* 2, 637--644. https://doi.org/10.1038/s41562-018-0399-z
- Project repository: https://osf.io/pfdyw/

The source archive contains both `.csv` and Stata `.dta` versions of the
D1--D8 aggregate tables.  The data builder only needs `D2 -
ReplicationSampleSizes.csv` and `D3 - ReplicationResults.dta`, but the full
bundle is retained so that its inputs and provenance can be independently
checked.  These source data remain subject to their original terms; the
repository's MIT license applies only to this repository's original code and
documentation.

## SHA-256 checksums

```text
e25e27c45226da35a353b73462a4d5f788b4db6d87fdcb49eff990120a4de3c5  D1 - OriginalStudies.csv
9ab3e7a37b5867dec478a72c2e006db98d3c509a648e1422ffb42ff3ee99e39d  D1 - OriginalStudies.dta
82229846c562fa05db4c1c93144d4aef87d16358d8256d722116b4385b3546f3  D2 - ReplicationSampleSizes.csv
e677f08acd03fc4352857fd76e89834b7cc6377107074efcf909747ef6ac1723  D2 - ReplicationSampleSizes.dta
6c81ea23d2144023f2f49c645a2e6b33055af80ec86a3b100f871e24382ad94f  D3 - ReplicationResults.csv
444bac5096f0322a9cc7c01ad9b24a9e8953cca746fcff98cc03afc988a93197  D3 - ReplicationResults.dta
204c4fa9f497af86539af04a4fe2a7b6805fb5ba0be516dc4ce4c16fa371f48a  D4 - MarketHoldings.csv
2dbbec73eb76446161ccacc4eaba1ea777cef1b134ebdcea54d41b43a6947c5e  D4 - MarketHoldings.dta
fd48317e57f7a78a1d69d62246391010e7b4a94e2a59f3ba9eb50a7e905abee4  D4 - MarketTransactions.csv
a27008bb67d034c57c2794062cb2e7dbb05f233b87febce6cf51a7311fc60f8b  D4 - MarketTransactions.dta
c0487cd9ba972bd36e82b932ac94d85194089af202a01c8a784e5565f73bb5e2  D5 - PreMarketSurvey.csv
1151b02cd478e116bc009bfd03e6664396b2d2762957a656cc1670301a34a314  D5 - PreMarketSurvey.dta
dd81f6b3ef876bb51e0b5428ce33a3ffed60cf64279e222c7ea3a5fce287adea  D6 - Demographics.csv
db5f16eee5175cc328dcf95c2393865e6f8e5a30c5632584452d27a60f48cc49  D6 - Demographics.dta
47239a3b8abcd06ac5fde0b275b3c3c095f206ff9ac264a252360f37526592e8  D6 - MeanPeerBeliefs.csv
3ed46e57c2ed61a8cda8ebd6a718eb89aea9e9f250ff06027b08c456e494a777  D6 - MeanPeerBeliefs.dta
d969e191a9533443c8b24dc15d4a651cb010b18a83fc9409ce496aced0f1dd12  D7 - Comparison.csv
cb68bb66ebdf773be4a15052b6e6a58bee7f8a67eb2d9a6ea1172b46a495c7a4  D7 - Comparison.dta
c149ac13f004d841db71d95cea2ec3309cbd3aac8c0945f44f9f2b836dc6c114  D8 - RobustnessTests.csv
90147af2bbf96575bfe16d30790548b03dc90de34aa66a4bc1c104e85f4f99ec  D8 - RobustnessTests.dta
```
