#!/bin/bash

set -eu -o pipefail

jobName=(iid_1 iid_2)
nblocks=(128 640 6400)

function writeJob {
    pers=$1
    rfs=$2
    efs=$3
    nblocks=$4
    jn=$5
    jobN=$jn.$nblocks.job
    echo -e "#!/bin/bash" > ./$jobN
    echo -e "#$ -cwd -m ea"  >> ./$jobN

    for per in ${pers[*]}
    do
        for rf in ${rfs[*]}
        do
            for ef in ${efs[*]}
            do
                printf "python3 run_uep_iid.py --iid_per  %s %s %s %s\n" $per $rf $ef $nblocks >> ./$jobN
                #python3 run_uep_iid.py $rf $ef $nblocks
            done
        done
    done
    qsub $jobN
}

for nbl in ${nblocks[*]}
do
    for jn in ${jobName[*]}
    do
        if [ $jn=iid_1 ]; then
            rfs=(1 5)
            efs=(1 2 4)
            pers=(0)
            writeJob $pers $rfs $efs $nbl $jn
        else
            rfs=(5)
            efs=(2)
            pers=(0.01 0.1 0.2 0.3)
            writeJob $pers $rfs $efs $nbl $jn
        fi
    done
done