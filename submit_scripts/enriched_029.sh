#!/bin/bash

date
startsecond=$(date +%s)

echo "I'm process id $$ on" `hostname`

echo "Starting the processing job"
echo "Argument line : " $@

echo "Starting cvmfs "
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh`
echo "cvfms ran"

i3env=/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/metaprojects/icetray/v1.9.2/env-shell.sh
echo "Will use i3 environment: " ${i3env}
script=/data/user/akatil/electron_neutrino/for_real/analysis_chain/extract_var_enriched_029.py
#script=/data/user/akatil/electron_neutrino/for_real/analysis_chain/enriched_029.py
echo "Will use script: " $script

DATASET=$1

echo "Dataset : "$DATASET

$i3env python $script --dataset $DATASET

date
endsecond=$(date +%s)
echo "End second: " $endsecond
echo "This job took : "`expr $endsecond - $startsecond`" s"
