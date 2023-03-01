#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
#SBATCH --account=free # The account name for the job.
basejobname="curves"

jobfile="curves.sh"

for momentum in 0.999 0.995 0.99 0.9; do
    for rate in 0.00001 0.0001 0.001 0.01 0.1 1 5; do 
        jobname=$basejobname-$momentum-$rate
        echo "Submitting job $jobname"

        ##outfile = output/curves-momentum.$momentum-rate.$rate.out

        export momentum;
        export rate;

        sbatch --job-name $jobname $jobfile
    done;
done

basejobname="curves_tanh"

jobfile="curves_tanh.sh"

for momentum in 0.999 0.995 0.99 0.9; do
    for rate in 0.00001 0.0001 0.001 0.01 0.1 1 5; do 
        jobname=$basejobname-$momentum-$rate
        echo "Submitting job $jobname"

        ##outfile = output/curves-momentum.$momentum-rate.$rate.out

        export momentum;
        export rate;

        sbatch --job-name $jobname $jobfile
    done;
done

basejobname="curves_relu"
jobfile="curves_relu.sh"

for momentum in 0.999 0.995 0.99 0.9; do
    for rate in 0.00001 0.0001 0.001 0.01 0.1 1 5; do 
        jobname=$basejobname-$momentum-$rate
        echo "Submitting job $jobname"

        ##outfile = output/curves-momentum.$momentum-rate.$rate.out

        export momentum;
        export rate;

        sbatch --job-name $jobname $jobfile
    done;
done

basejobname="curves_leaky"
jobfile="curves_leaky.sh"

for momentum in 0.999 0.995 0.99 0.9; do
    for rate in 0.00001 0.0001 0.001 0.01 0.1 1 5; do 
        jobname=$basejobname-$momentum-$rate
        echo "Submitting job $jobname"

        ##outfile = output/curves-momentum.$momentum-rate.$rate.out

        export momentum;
        export rate;

        sbatch --job-name $jobname $jobfile
    done;
done

basejobname="curves_no_fsigmoid"
jobfile="curves_no_fsigmoid.sh"

for momentum in 0.999 0.995 0.99 0.9; do
    for rate in 0.00001 0.0001 0.001 0.01 0.1 1 5; do 
        jobname=$basejobname-$momentum-$rate
        echo "Submitting job $jobname"

        ##outfile = output/curves-momentum.$momentum-rate.$rate.out

        export momentum;
        export rate;

        sbatch --job-name $jobname $jobfile
    done;
done

basejobname="curves_no_hsigmoid"
jobfile="curves_no_hsigmoid.sh"

for momentum in 0.999 0.995 0.99 0.9; do
    for rate in 0.00001 0.0001 0.001 0.01 0.1 1 5; do 
        jobname=$basejobname-$momentum-$rate
        echo "Submitting job $jobname"

        ##outfile = output/curves-momentum.$momentum-rate.$rate.out

        export momentum;
        export rate;

        sbatch --job-name $jobname $jobfile
    done;
done

echo "All jobs submited"

