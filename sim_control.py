import os
import math
import time
import numpy as np
import pyproj
import logging
import lgsvl
from simple_pid import PID

delta_steer_per_iter = 0.005

# Stanley controller parameters
k_e = 0.33
k_v = 0.01

utm_zone = 10
my_proj = pyproj.Proj("+proj=utm +zone="+str(utm_zone)+", +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

def lonlat_to_xy(lon, lat):
    '''convert lon/lat (in degrees) to x/y native map projection coordinates (in meters)'''
    x, y = my_proj(lon, lat)
    return x, y


def xy_to_lonlat(x, y):
    lon, lat = my_proj(x, y, inverse=True)
    return lon, lat


def get_bearing(lat1, lon1, lat2, lon2):
    dLon = lon2 - lon1
    x = np.cos(np.radians(lat2)) * np.sin(np.radians(dLon))
    y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(dLon))
    brng = np.arctan2(x, y)
    brng = (np.degrees(brng) + 360) % 360
    return brng


def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2.0*np.pi
    while angle < -np.pi:
        angle += 2.0*np.pi
    return angle


def on_left(pt, pt1, pt2):
    return (pt2[0]-pt1[0])*(pt[1]-pt1[1]) - (pt2[1]-pt1[1])*(pt[0]-pt1[0]) > 0


def distance_to_line(pt, pt1, pt2):
    slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    b = pt2[1] - slope*pt2[0]
    c = pt[1] + (pt[0]/slope)
    dist = abs(pt[1] - slope*pt[0] - b) / math.sqrt(1+slope*slope)
    return dist


def distance_to_point_and_angle(pt, line_pt, line_angle_radian):
    slope = np.sin(line_angle_radian)
    b = line_pt[1] - slope*line_pt[0]
    c = pt[1] + (pt[0]/slope)
    dist = abs(pt[1] - slope*pt[0] - b) / math.sqrt(1+slope*slope)
    return dist


def calc_crosstrack_error(waypt, waypts):
    curr_idx = -1
    for i in range(len(waypts)):
        if abs(waypt[0] - waypts[i][0]) < 1e-5:
            curr_idx = i
            break
    if curr_idx == -1:
        print("Waypoint matching error!")
        raise
    if curr_idx == len(waypts) - 1:
        pt1 = [waypts[curr_idx - 1][1], waypts[curr_idx - 1][2]]
        pt2 = [waypts[curr_idx][1], waypts[curr_idx][2]]
    else:
        pt1 = [waypts[curr_idx][1], waypts[curr_idx][2]]
        pt2 = [waypts[curr_idx + 1][1], waypts[curr_idx + 1][2]]
    pt = [waypt[1], waypt[2]]
    crosstrack_error = distance_to_line(pt, pt1, pt2)
    return crosstrack_error


class VehicleState:
    time = 0.
    speed = 0.
    position = None
    attitude = None


class SimControl:
    def __init__(self, frame_dir, npc_traj,
            sim_map="Achilles-AdversarialNPC-SanFranciscoSingleBox",
            sim_veh="Achilles-AdversarialNPC-Lincoln2017MKZ",
            sim_speed=20.0):

        self.sim = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181)
        self.adc_name = sim_veh
        if self.sim.current_scene == sim_map:
            self.sim.reset()
        else:
            self.sim.load(sim_map, seed=100)  # fix the seed

        spawns = self.sim.get_spawn()
        state = lgsvl.AgentState()
        state.transform.position = spawns[0].position
        state.transform.rotation = spawns[0].rotation
        forward = lgsvl.utils.transform_to_forward(spawns[0])  # forward unit vector
        up = lgsvl.utils.transform_to_up(spawns[0])  # up unit vector
        right = lgsvl.utils.transform_to_right(spawns[0])  # right unit vector
        lateral_shift = -4.0
        forward_shift = 0.
        state.transform.position = spawns[0].position + lateral_shift * right + forward_shift * forward
        state.velocity = sim_speed * forward
        self.adc = self.sim.add_agent(self.adc_name, lgsvl.AgentType.EGO, state)
        sensors = self.adc.get_sensors()
        self.front_cam = None
        # self.free_cam = None
        for s in sensors:
            # if s.name == "3rd Person View":
            #     s.enabled = True
            #     self.free_cam = s
            if s.name == "Main Camera":
                s.enabled = True
                self.front_cam = s
            else:
                s.enabled = False

        self.npc_name = "Sedan"
        npc_trans = self.sim.map_from_gps(
                latitude=npc_traj[0]['latitude'],
                longitude=npc_traj[0]['longitude'],
                altitude=11.0) # npc_traj[0]['altitude'])
        npc_trans.rotation = lgsvl.Vector(
                npc_traj[0]['roll'], npc_traj[0]['yaw'], npc_traj[0]['pitch'])

        layer_mask = 0
        layer_mask |= 1 << 0  # 0 -- road
        hit = self.sim.raycast(npc_trans.position, -up, layer_mask)
        npc_trans.position = hit.point

        npc_forward = lgsvl.utils.transform_to_forward(npc_trans)
        npc_state = lgsvl.AgentState()
        npc_state.transform.position = npc_trans.position
        npc_state.transform.rotation = npc_trans.rotation
        npc_state.velocity = npc_traj[0]['speed'] * npc_forward

        self.npc = self.sim.add_agent(self.npc_name, lgsvl.AgentType.NPC, npc_state)
        wps = []
        for i in range(1, len(npc_traj)):
            npc_trans = self.sim.map_from_gps(
                    latitude=npc_traj[i]['latitude'],
                    longitude=npc_traj[i]['longitude'],
                    altitude=11.0) # npc_traj[i]['altitude'])
            layer_mask = 0
            layer_mask |= 1 << 0  # 0 -- road
            hit = self.sim.raycast(npc_trans.position, -up, layer_mask)
            npc_trans.position = hit.point
            npc_trans.rotation = lgsvl.Vector(
                    npc_traj[i]['roll'], npc_traj[i]['yaw'], npc_traj[i]['pitch'])
            
            # if i < 190:
            #     npc_traj[i]['speed'] = 28 # TODO
            # elif i < 210:
            #     npc_trans.position.x -= 3.5
            #     npc_traj[i]['speed'] = 28 # TODO
            #     tmp = npc_trans.position.z
            # elif i < 210+10:
            #     npc_trans.position.x -= 3.5# / 10 * (210+10-i)
            #     npc_traj[i]['speed'] = 11
            #     dy = npc_trans.position.z - tmp
            #     tmp = npc_trans.position.z
            # else: # just left box
            #     npc_trans.position.x -= 3.5# / 10 * (210+10-i)
            #     npc_traj[i]['speed'] = 25
            #     tmp -= dy / 28 * 25
            #     npc_trans.position.z = tmp

            wp = lgsvl.DriveWaypoint(npc_trans.position, npc_traj[i]['speed'],
                    npc_trans.rotation, 0, 0)
            wps.append(wp)
        self.npc.follow(wps)
        self.adc.on_collision(self.on_collision)

        # Car control knob, steering [-1, 1] +: right, -: left
        self.ctrl = lgsvl.VehicleControl()
        self.ctrl.throttle = 0.0
        self.ctrl.steering = 0.0
        self.ctrl.braking = 0.0
        self.ctrl.reverse = False
        self.ctrl.handbrake = False
        self.adc.apply_control(self.ctrl, True)

        self.pid = PID(1.0, 0.1, 0.01, setpoint=sim_speed)
        self.pid.sample_time = 0.01
        self.pid.output_limits = (-1, 1)

        ctrl_freq = 20
        self.ctrl_period = 1.0/ctrl_freq
        self.sim_time = 0.

        self.last_steer = 0.
        self.candidate_plans = []
        self.curr_plan = 0
        self.default_plan = 0
        self.last_npc_dist2adc = float('-inf')
        self.lane_change_threshold = 130

        self.frame_id = 0
        self.frame_dir = frame_dir
        if not os.path.exists(self.frame_dir):
            print(f"Frame folder does not exist, failed to run simulation")
            raise

    def set_candidate_plans(self, plans):
        self.candidate_plans = plans
        self.curr_plan = 0  # always assume 1st plan as the default one
        self.default_plan = 0

    def on_collision(self, agent1, agent2, contact):
        name1 = "STATIC OBSTACLE" if agent1 is None else agent1.name
        name2 = "STATIC OBSTACLE" if agent2 is None else agent2.name
        logging.info("Collision just happened!")
        print("Collision: {} collided with {}".format(name1, name2), "at", contact)

    def dist_to_plan(self, x, y, plan_id):
        return np.sqrt(np.min(np.sum((
          np.array([x, y]) - np.array(self.candidate_plans[plan_id])[:, 1:])**2, axis=1)))

    def need_lane_change(self):
        if not self.is_lane_change_done():
            return False
        npc_position = self.sim.map_to_gps(self.npc.state.transform)
        npc_lon = npc_position[1]
        npc_lat = npc_position[0]
        npc_x, npc_y = lonlat_to_xy(npc_lon, npc_lat)
        adc_position = self.sim.map_to_gps(self.adc.state.transform)
        adc_lon = adc_position[1]
        adc_lat = adc_position[0]
        adc_x, adc_y = lonlat_to_xy(adc_lon, adc_lat)
        adc_bearing = (adc_position[5] - 90 + 360) % 360
        # check if default path is clear
        adc2npc_bearing = get_bearing(npc_lat, npc_lon, adc_lat, adc_lon)
        relative_bearing = (adc2npc_bearing - adc_bearing + 360) % 360
        need_change = False
        npc_dist2adc = np.sqrt(np.sum((np.array([npc_x, npc_y]) - np.array([adc_x, adc_y]))**2))
        print("NPC-Ego dist:", npc_dist2adc)
        # 1. check if NPC blocking current traj
        npc_front = False
        if relative_bearing < 90 or relative_bearing > 270:
            print("NPC is in front of us")
            npc_front = True
            npc_dist2curr_plan = self.dist_to_plan(npc_x, npc_y, self.curr_plan)
            npc_curr_plan_overlap = False
            if npc_dist2curr_plan < 1.5:
                npc_curr_plan_overlap = True
            if npc_curr_plan_overlap and npc_dist2adc < self.last_npc_dist2adc and npc_dist2adc < self.lane_change_threshold:
                print("Need changing lane due to blocking")
                need_change = True
        # 2. check if NPC blocking default traj, change back to default otherwise
        if self.curr_plan != self.default_plan:
            npc_dist2default_plan = self.dist_to_plan(npc_x, npc_y, self.default_plan)
            npc_default_plan_overlap = False
            if npc_dist2default_plan < 1.5:
                npc_default_plan_overlap = True
            print("NPC-default dist:", npc_dist2default_plan)
            if not npc_front:
                print("Need changing back to default lane due to no front NPC")
                need_change = True
            elif not npc_default_plan_overlap or npc_dist2adc > self.lane_change_threshold:
                print("Need changing back to default lane due to far front NPC")
                need_change = True
        self.last_npc_dist2adc = npc_dist2adc
        return need_change

    def is_lane_change_done(self):
        adc_position = self.sim.map_to_gps(self.adc.state.transform)
        lon = adc_position[1]
        lat = adc_position[0]
        x, y = lonlat_to_xy(lon, lat)
        dist2plan = self.dist_to_plan(x, y, self.curr_plan)
        if dist2plan < 1.0:
            return True
        else:
            return False

    def next_yuv_frame(self, save=True):
        frontFramePath = os.path.join(self.frame_dir, "front_frame_" + str(self.frame_id) + ".png")
        if save:
            self.front_cam.save(frontFramePath, compression=3)
        ret_frame_id = self.frame_id
        self.frame_id += 1
        return ret_frame_id

    def get_adc_state(self):
        adcState = VehicleState()
        adcState.time = self.sim_time
        adcState.speed = self.adc.state.speed
        adcState.position = self.sim.map_to_gps(self.adc.state.transform)
        adcState.attitude = [self.adc.state.rotation.x, self.adc.state.rotation.y, self.adc.state.rotation.z]
        return adcState

    def get_npc_state(self):
        npcState = VehicleState()
        npcState.time = self.sim_time
        npcState.speed = self.npc.state.speed
        npcState.position = self.sim.map_to_gps(self.npc.state.transform)
        npcState.attitude = [self.npc.state.rotation.x, self.npc.state.rotation.y, self.npc.state.rotation.z]
        return npcState

    def apply_control(self, throttle, steering):
        self.ctrl.throttle = throttle
        self.ctrl.steering = steering
        self.adc.apply_control(self.ctrl, True)  # sticky control
        self.sim.run(time_limit=self.ctrl_period, time_scale = 2)
        self.sim_time += self.ctrl_period

    def lateral_control(self, waypt, v, yaw):
        # Ref: https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf
        yaw_path = np.arctan2(
            self.candidate_plans[self.curr_plan][-1][2] - self.candidate_plans[self.curr_plan][0][2],
            self.candidate_plans[self.curr_plan][-1][1] - self.candidate_plans[self.curr_plan][0][1])
        yaw_diff = normalize_angle(yaw_path - yaw)
        closest_idx = np.argmin(
                np.sum((np.array(waypt[1:]) - np.array(self.candidate_plans[self.curr_plan])[:, 1:])**2, axis=1))
        crosstrack_error = distance_to_point_and_angle(
                self.candidate_plans[self.curr_plan][closest_idx][1:], waypt[1:], yaw)
        yaw_crosstrack = np.arctan2(
                waypt[2] - self.candidate_plans[self.curr_plan][0][2],
                waypt[1] - self.candidate_plans[self.curr_plan][0][1])
        yaw_path2ct = normalize_angle(yaw_path - yaw_crosstrack)
        if yaw_path2ct > 0:
            crosstrack_error = abs(crosstrack_error)
        else:
            crosstrack_error = -abs(crosstrack_error)
        yaw_diff_crosstrack = np.arctan(k_e*crosstrack_error/(k_v + v))
        steer = normalize_angle(yaw_diff + yaw_diff_crosstrack)
        steer = -np.clip(steer, -1., 1.)
        steer_limited = np.clip(steer,
                self.last_steer - delta_steer_per_iter,
                self.last_steer + delta_steer_per_iter)
        self.last_steer = steer_limited
        return steer_limited

    def longitudinal_control(self, waypt, v, yaw):
        throttle = self.pid(v)
        return throttle

    def run_control(self, curr_waypt):
        sim_state = self.get_adc_state()
        speed = sim_state.speed
        yaw = np.radians(sim_state.position[5])
        last_steering = self.ctrl.steering

        s_2 = time.time()
        steering = self.lateral_control(curr_waypt, speed, yaw)
        throttle = self.longitudinal_control(curr_waypt, speed, yaw)
        # print('_control: {}'.format(time.time()-s_2))
        s_2 = time.time()
        self.apply_control(throttle, steering)
        # print('_apply_control: {}'.format(time.time()-s_2))
        sim_state = self.get_adc_state()
        s_2 = time.time()
        if self.need_lane_change():
            self.curr_plan = 1 - self.curr_plan
        # print('_need_lane_change: {}'.format(time.time()-s_2))
        return sim_state
