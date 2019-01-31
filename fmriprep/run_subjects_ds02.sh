#!/bin/bash
for i in {2..15}
do
   subject=$(printf "%02d" $i)
   echo "Running subject $subject"

   singularity run ~/fmriprep-1.2.6.simg ~/data/bias_task/fmriprep_data/ ~/data/bias_task/derivatives participant --no-submm-recon --fs-license-file ~/license.txt --output-space T1w template fsaverage --nthreads 12 -w /home/hollander/workflow_folders/ --force-syn --participant-label $subject
done
