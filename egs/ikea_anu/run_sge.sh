#!/usr/bin/env bash
#$ -wd /home/jdjones/data/output/grid_logs
#$ -V
#$ -N ikea_anu_pipeline
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M jdjones@jhu.edu
#$ -m e
#$ -l ram_free=15G,mem_free=15G,gpu=1,hostname=b1[123456789]|c0[12345678]|c1[123456789]

set -ue

conda activate kinemparse
cd /home/jdjones/repo/kinemparse/egs/ikea_anu

./run.sh "$@"
