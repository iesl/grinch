# Grinch #

Code for [Scalable Hierarchical Clustering with Tree Grafting](https://dl.acm.org/citation.cfm?id=3330929). Nicholas Monath*, Ari Kobren*, Akshay Krishnamurthy, Michael Glass, Andrew McCallum. KDD 2019


## Setup ##

Set environment variables (run this in every session):

```
source bin/setup.sh
```

Install maven if you don't already have it installed:

```
./bin/util/install_mvn.sh
```

You may need to set `JAVA_HOME` and `JAVA_HOME_8`.

Build Scala code:

```
./bin/build.sh
```

Download data:

```
./bin/download_data.sh
```

## Run ##

Run methods on the synthetic separated data:

```
# Grinch alg. w/o approximations
sh bin/cslink/synth/full.sh

# Grinch alg. w/ graft cap
sh bin/cslink/synth/graft_cap.sh

# Grinch alg. w/ single elimination grafting
sh bin/cslink/synth/single_elimination.sh

# Remaining scripts cover other rows in Table 1 in the paper.
```

Run methods on ALOI: 

```
# Greedy baseline
sh bin/avglink/aloi/run_greedy.sh

# Rotate online baseline
sh bin/avglink/aloi/run_rotate.sh

# GRINCH
sh bin/avglink/aloi/run_graft_restruct.sh
```

The `launch_slurm.sh` script can be used to submit a job to the slurm 
cluster manager:

```
sh bin/launch_slurm.sh <script_to_run> <dataset_name> <alg_name> <run_id=1> <threads=4> <mem=20000>
sh bin/launch_slurm.sh bin/cslink/synth/full.sh Synth Grinch 
```

## Notes ##

  - The ALOI scripts are set up to run on a machine with about 8 cores and 60GB of memory.
  - You'll need perl installed on your system to run experiment shell scripts as is. perl is used to shuffle the data. If you can't run perl, you can change this to another shuffling method of your choice.
  - The scripts in this project use environment variables set in the setup script. You'll need to source this set up script in each shell session running this project.
  - Java Version 1.8 and Scala 2.11.7 are used in this project. Java 1.8 must be installed on your system. It is not necessary to have Scala installed.

## Citation ##

```
@inproceedings{Monath:2019:SHC:3292500.3330929,
 author = {Monath, Nicholas and Kobren, Ari and Krishnamurthy, Akshay and Glass, Michael R. and McCallum, Andrew},
 title = {Scalable Hierarchical Clustering with Tree Grafting},
 booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
 series = {KDD '19},
 year = {2019},
 isbn = {978-1-4503-6201-6},
 location = {Anchorage, AK, USA},
 pages = {1438--1448},
 numpages = {11},
 url = {http://doi.acm.org/10.1145/3292500.3330929},
 doi = {10.1145/3292500.3330929},
 acmid = {3330929},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {clustering, hierarchical clustering},
} 
```

## License

Apache License, Version 2.0

## Questions / Comments / Bugs / Issues

Please contact Nicholas Monath (nmonath@cs.umass.edu).

Also, please contact me for access to the data.

