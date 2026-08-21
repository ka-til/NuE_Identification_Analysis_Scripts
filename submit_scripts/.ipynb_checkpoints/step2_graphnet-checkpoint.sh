#!/bin/bash

date
startsecond=$(date +%s)

echo "I'm process id $$ on" `hostname`

echo "Starting the processing job"
echo "Argument line : " $@

#source /home/akatil/graphnet_venv/bin/activate
#echo "Will use virtual environment "

echo "Starting cvmfs "
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh`
echo "cvfms ran"

#source /home/akatil/graphnet_venv/bin/activate
#echo "Will use virtual environment "

i3env=/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/metaprojects/icetray/v1.9.2/env-shell.sh
echo "Will use i3 environment: " ${i3env}
#source $i3env

source /home/akatil/graphnet_venv/bin/activate #/data/user/akatil/electron_neutrino/for_real/venv/graphnet_gpu/bin/activate
echo "Will use virtual environment "

export PYTHONPATH=/home/akatil/graphnet_venv/lib/python3.11/site-packages:$PYTHONPATH
export PYTHONPATH=/home/akatil/graphnet_venv/lib:$PYTHONPATH

script=/data/user/akatil/electron_neutrino/for_real/analysis_chain/step2_graphnet_reco.py
echo "Will use script: " $script

DATASET=$1

echo "Dataset : "$DATASET

$i3env python $script --dataset $DATASET --num-workers 16

date
endsecond=$(date +%s)
echo "End second: " $endsecond
echo "This job took : "`expr $endsecond - $startsecond`" s"
