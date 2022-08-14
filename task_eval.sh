#!/bin/bash

input_file=$(realpath ./npc.json)
logdir=$(realpath ./data)

mkdir -p ${logdir}

ret=`python task_eval.py --conf task_conf.yml --task 0 --logdir ${logdir} --input_traj ${input_file}`

ffmpeg -r 20 -i ${logdir}/frames/front_frame_%d.png -vcodec libx264 -pix_fmt yuv420p -vf "scale=iw/2:ih/2" ${logdir}/sim_view.mp4 -y
