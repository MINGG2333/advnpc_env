#!/usr/bin/env python

import os
import sys
import json
import yaml
import time
import logging
import cv2
import argparse

from sim_control import SimControl, lonlat_to_xy

SIM_FRAMES = 400

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('--conf', type=str, help='Config file')
parser.add_argument('--task', type=int, help='Task id, e.g., 0')
parser.add_argument('--input_traj', type=str, help='Adversarial NPC traj file submitted by player')
parser.add_argument('--logdir', type=str, help='Folder to store evaluation results')


def get_task_conf(conf_file, task_id):
    conf = None
    with open(conf_file, 'r') as f:
        all_conf = yaml.load(f, Loader=yaml.FullLoader)
        if not task_id in all_conf['task_id']:
            failed("task id is invalid: " + str(task_id))
            return
        conf = all_conf[task_id]
    return conf


def parse_pose(pose_file, plot=False):
    with open(pose_file, 'r') as f:
        waypts_lla = json.load(f)

    pts = []
    for pt in waypts_lla:
        x, y = lonlat_to_xy(pt['longitude'], pt['latitude'])
        pts.append([pt['timestamp'], x, y])
    return pts


def parse_npc_traj(traj_file):
    with open(traj_file, 'r') as f:
        traj = json.load(f)
    return traj


def task_eval(logdir, adc_plan_traj_files, attack_npc_traj_file, **kwargs):
    sim_map = kwargs['map']

    adc_plans = []
    for i in range(len(adc_plan_traj_files)):
        plan = parse_pose(adc_plan_traj_files[i])
        adc_plans.append(plan)
    npc_traj = parse_npc_traj(attack_npc_traj_file)

    frame_dir = os.path.join(logdir, 'frames')
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)

    sim_ctrl = SimControl(frame_dir, npc_traj, sim_map=sim_map)
    sim_ctrl.set_candidate_plans(adc_plans)
    time.sleep(1)
    adc_state = sim_ctrl.get_adc_state()
    adc_poses = []
    adc_poses.append({
        "timestamp": round(adc_state.time, 2),
        "latitude": adc_state.position[0],
        "longitude": adc_state.position[1],
        "altitude": adc_state.position[4],
        "roll": adc_state.attitude[0],
        "pitch": adc_state.attitude[2],
        "yaw": adc_state.attitude[1],
        "speed": adc_state.speed})

    npc_state = sim_ctrl.get_npc_state()
    npc_poses = []
    npc_poses.append({
        "timestamp": round(npc_state.time, 2),
        "latitude": npc_state.position[0],
        "longitude": npc_state.position[1],
        "altitude": npc_state.position[4],
        "roll": npc_state.attitude[0],
        "pitch": npc_state.attitude[2],
        "yaw": npc_state.attitude[1],
        "speed": npc_state.speed})
    frame_count = 0
    s_ = time.time()
    while True:
        print("Time {:.2f}".format(sim_ctrl.sim_time))
        frame_id = sim_ctrl.next_yuv_frame(save=False)

        # s_2 = time.time()
        # Serve as localization
        adc_state = sim_ctrl.get_adc_state()
        x, y = lonlat_to_xy(adc_state.position[1], adc_state.position[0])
        curr_waypt = [adc_state.time, x, y]
        # print('lonlat_to_xy: {}'.format(time.time()-s_2))

        # s_2 = time.time()
        adc_state = sim_ctrl.run_control(curr_waypt)
        # print('run_control: {}'.format(time.time()-s_2))
        adc_poses.append({
            "timestamp": round(adc_state.time, 2),
            "latitude": adc_state.position[0],
            "longitude": adc_state.position[1],
            "altitude": adc_state.position[4],
            "roll": adc_state.attitude[0],
            "pitch": adc_state.attitude[2],
            "yaw": adc_state.attitude[1],
            "speed": adc_state.speed})
        npc_state = sim_ctrl.get_npc_state()
        npc_poses.append({
            "timestamp": round(npc_state.time, 2),
            "latitude": npc_state.position[0],
            "longitude": npc_state.position[1],
            "altitude": npc_state.position[4],
            "roll": npc_state.attitude[0],
            "pitch": npc_state.attitude[2],
            "yaw": npc_state.attitude[1],
            "speed": npc_state.speed})
        # print('tick_time: {}'.format(time.time()-s_))
        frame_count += 1
        if frame_count >= SIM_FRAMES:
            break

    print('total_time: {}'.format(time.time()-s_))
    # Save ADC pose
    pose_file = os.path.join(logdir, 'adc_pose.json')
    with open(pose_file, 'w') as f:
        json.dump(adc_poses, f, indent=2)

    # Save NPC pose
    pose_file = os.path.join(logdir, 'npc_pose.json')
    with open(pose_file, 'w') as f:
        json.dump(npc_poses, f, indent=2)


def main():
    args = parser.parse_args()
    challenge_dir = os.path.dirname(os.path.abspath(args.conf))
    conf = get_task_conf(args.conf, args.task)
    logdir = os.path.abspath(args.logdir)

    logging.basicConfig(filename=os.path.join(logdir, "simulation.log"),
            level=logging.INFO, format="[%(asctime)s] %(message)s")

    left_plan_file = os.path.join(challenge_dir, "traj", "ego_left.json")
    right_plan_file = os.path.join(challenge_dir, "traj", "ego_right.json")
    attack_npc_traj_file = args.input_traj

    task_eval(logdir, [left_plan_file, right_plan_file], attack_npc_traj_file, **conf['args_eval'])


if __name__ == "__main__":
    main()
