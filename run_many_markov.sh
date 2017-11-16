#!/bin/bash

set -eu -o pipefail

jobName=(mar_2 mar_3 mar_4)
nblocks=(6400)

function writeJob {
    pis=$4
    rfs=$1
    efs=$2
    nblocks=$3
    enb=$5
    jn=$6
    jobN=$jn.$enb.job
    echo -e "#!/bin/bash" > ./$jobN
    echo -e "#$ -cwd -m ea" >> ./$jobN
    echo -e "set -eu -o pipefail" >> ./$jobN
    printf "Delete this file to end the job\n" > ./$jobN.check
    printf "file=%s.check\n" $jobN >> $jobN
    printf "while [ -f \"\$file\" ]; do\n" >> $jobN

    for pib in ${pis[*]}
    do
        for rf in ${rfs[*]}
        do
            for ef in ${efs[*]}
            do
                printf "   python3 run_uep_markov.py " >> ./$jobN
                if [ $pib == 100 ]; then
                    k_min=600
                    k_max=2300
                    printf " --kmin=%s --kmax=%s --overhead=0.3 " $k_min $k_max >> ./$jobN
                    #python3 run_uep_iid.py $rf $ef $nblocks-
                fi
                if [ $pib == 10 ]; then
                    k_min=6000
                    k_max=8000
                    printf " --kmin=%s --kmax=%s --overhead=0.3 " $k_min $k_max >> ./$jobN
                fi
                printf "%s %s %s %s %s\n" $rf $ef $nblocks $pib $enb >> ./$jobN                    
            done
        done
    done
    printf "done\n" >> $jobN
    qsub $jobN
}

for nbl in ${nblocks[*]}
do
    for jn in ${jobName[*]}
    do
        if [ $jn == mar_1 ]; then
            echo -e $jn
            rfs=(1 3 5)
            efs=(1)
            pib=(inf)
            enb=(1)
            writeJob $rfs $efs $nbl $pib $enb $jn
        else
            if [ $jn == mar_2 ]; then
                rfs=(1 5)
                efs=(1)
                pib=(100 10)
                enb=(5)
                writeJob $rfs $efs $nbl $pib $enb $jn
            else
                if [ $jn == mar_3 ]; then
                    rfs=(1 5)
                    efs=(1)
                    pib=(100 10)
                    enb=(10)
                    writeJob $rfs $efs $nbl $pib $enb $jn
                else
                    rfs=(1 5)
                    efs=(1)
                    pib=(100 10)
                    enb=(50)
                    writeJob $rfs $efs $nbl $pib $enb $jn
                fi
            fi
        fi
    done
done