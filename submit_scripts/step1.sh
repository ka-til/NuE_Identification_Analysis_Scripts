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
script=/data/user/akatil/electron_neutrino/for_real/analysis_chain/step1_processing.py
echo "Will use script: " $script

INTERACTION=$1
DATASET=$2

echo "Interaction: " $INTERACTION
echo "Dataset : "$DATASET

OUTDIR=/data/user/akatil/electron_neutrino/for_real/dataset_complete/

echo "OUTPUT FILE DIR  : "$OUTDIR

$i3env python $script --interaction $INTERACTION --dataset $DATASET --outdir $OUTDIR

date
endsecond=$(date +%s)
echo "End second: " $endsecond
echo "This job took : "`expr $endsecond - $startsecond`" s"
