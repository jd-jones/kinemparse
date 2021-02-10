#!/usr/bin/env bash
#$ -wd /home/jdjones/data/output
#$ -V
#$ -N ikea_anu_pipeline
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M jdjones@jhu.edu
#$ -m e
#$ -l ram_free=10G,mem_free=10G

set -ue

conda activate kinemparse

cd /home/jdjones/repo/kinemparse/egs/ikea_anu
./run.sh --start_at=1 --stop_after=1
# ./videos_to_frames.sh
