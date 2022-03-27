from queue import PriorityQueue
import math as m
from .util import normalize_angle
import numpy as np
from pypoman import intersect_polygons, plot_polygon


class VehicleConfig:

    def __init__(self) -> None:
        self.length = 3.5
        self.width = 2
        self.baselink_to_front = 3
        self.baselink_to_rear = self.length - self.baselink_to_front
        self.wheel_base = 2.5
        self.lf = 1.5
        self.lr = self.wheel_base - self.lf
        self.max_front_wheel_angle = 0.6  # rad
        self.min_radius = self.wheel_base/m.tan(self.max_front_wheel_angle)
        self.T = 0.15  # discreate time
        self.max_acc = 5
        self.max_v = 5
        self.max_steer_rate = 20

class GridMap:
    def __init__(self, world_w, world_h) -> None:
        self.world_width = world_w
        self.world_height = world_h
        self.map_w = int(world_w/SE2State.xy_resolution)
        self.map_h = int(world_h/SE2State.xy_resolution)
        self.headings = int(m.pi*2/SE2State.heading_resolution)

    def generate_map(self):
        self.default_map = [[[SE2State(0, 0, 0)for i in range(self.headings)]
                             for i in range(self.map_h)] for i in range(self.map_w)]
        return self.default_map


class SE2State:
    xy_resolution = 0.15

    heading_resolution = 2*m.pi/70

    move_distance = 0.15
    T = 0.15

    def __init__(self, x, y, heading, cfg=VehicleConfig()):
        self.x = x
        self.y = y
        self.heading = normalize_angle(heading)
        self.v = 0
        self.a = 0
        self.steer = 0
        self.index = -1
        self.x_index = int(x/SE2State.xy_resolution)
        self.y_index = int(y/SE2State.xy_resolution)
        self.heading_index = int(
            (self.heading+m.pi)/SE2State.heading_resolution)
        self.visited = False
        self.direction_index = -1
        self.parent = [-1, -1, -1]
        self.cost_to_hear = 9999999
        self.cost_to_goal = 9999999
        # self.move_angle = SE2State.move_distance/cfg.min_radius
        self.move_angle = SE2State.heading_resolution


    def __lt__(self, other):
        return self.cost() < other.cost()

    def get_index(self):
        return [self.x_index, self.y_index, self.heading_index]

    def cost_to_state(self, state):
        x_error = state.x - self.x
        y_error = state.y - self.y
        return m.sqrt(x_error*x_error+y_error*y_error)

    def cost(self) -> float:
        return (self.cost_to_hear+self.cost_to_goal)*10

    def get_next_states(self):
        angle_plus = self.heading + self.move_angle
        angle_minus = self.heading - self.move_angle

        forward_state = SE2State(self.x+SE2State.move_distance*m.cos(self.heading),
                                 self.y+SE2State.move_distance *
                                 m.sin(self.heading),
                                 self.heading)
        forward_state.direction_index = 2
        backward_state = SE2State(self.x-SE2State.move_distance*m.cos(self.heading),
                                  self.y-SE2State.move_distance *
                                  m.sin(self.heading),
                                  self.heading)
        backward_state.direction_index = 5

        fl_state = SE2State(self.x+SE2State.move_distance*m.cos(angle_plus),
                            self.y+SE2State.move_distance*m.sin(angle_plus),
                            angle_plus)

        fl_state.direction_index = 1
        fr_state = SE2State(self.x+SE2State.move_distance*m.cos(angle_minus),
                            self.y+SE2State.move_distance*m.sin(angle_minus),
                            angle_minus)
        fr_state.direction_index = 3
        bl_state = SE2State(self.x-SE2State.move_distance*m.cos(angle_minus),
                            self.y-SE2State.move_distance*m.sin(angle_minus),
                            angle_minus)
        bl_state.direction_index = 4

        br_state = SE2State(self.x-SE2State.move_distance*m.cos(angle_plus),
                            self.y-SE2State.move_distance*m.sin(angle_plus),
                            angle_plus)
        br_state.direction_index = 6

        return [forward_state, backward_state, fl_state, fr_state, bl_state, br_state]


def generate_vehicle_vertices(state: SE2State, vehicle_config=VehicleConfig(), base_link=False):
    x = state.x
    y = state.y
    heading = state.heading
    L = vehicle_config.length
    W = vehicle_config.width
    b_to_f = vehicle_config.baselink_to_front
    b_to_r = vehicle_config.baselink_to_rear

    vertice_x = []
    vertice_y = []
    if(base_link):
        vertice_x = [x + b_to_f*m.cos(heading) - W/2*m.sin(heading),
                     x + b_to_f*m.cos(heading) + W/2*m.sin(heading),
                     x - b_to_r*m.cos(heading) + W/2*m.sin(heading),
                     x - b_to_r*m.cos(heading) - W/2*m.sin(heading)]

        vertice_y = [y + b_to_f*m.sin(heading) + W/2*m.cos(heading),
                     y + b_to_f*m.sin(heading) - W/2*m.cos(heading),
                     y - b_to_r*m.sin(heading) - W/2*m.cos(heading),
                     y - b_to_r*m.sin(heading) + W/2*m.cos(heading)]
    else:
        vertice_x = [x + L/2*m.cos(heading) - W/2*m.sin(heading),
                     x + L/2*m.cos(heading) + W/2*m.sin(heading),
                     x - L/2*m.cos(heading) + W/2*m.sin(heading),
                     x - L/2*m.cos(heading) - W/2*m.sin(heading)]

        vertice_y = [y + L/2*m.sin(heading) + W/2*m.cos(heading),
                     y + L/2*m.sin(heading) - W/2*m.cos(heading),
                     y - L/2*m.sin(heading) - W/2*m.cos(heading),
                     y - L/2*m.sin(heading) + W/2*m.cos(heading)]

    V = np.vstack((vertice_x, vertice_y)).T

    return V

def collsion(state: SE2State, obstacles, min_x=0, min_y=0, max_x=999, max_y=999) -> bool:
    host_vehicle = generate_vehicle_vertices(state, base_link=True)
    for vert in host_vehicle:
        x = vert[0]
        y = vert[1]
        if(x > max_x or x < min_x or y < min_y or y > max_y):
            return True
    for obs in obstacles:
        
        intersect_list = intersect_polygons(host_vehicle, obs)
        if not intersect_list:
            continue
        else:
            return True
    return False


def back_track(close_list, map):
    if(len(close_list) < 1):
        print('empty close list')
        return
    end = close_list[-1]
    path = []
    while end.parent != [-1, -1, -1]:
        path += [end]
        parent = end.parent
        end = map[parent[0]][parent[1]][parent[2]]
    path += [end]
    return path[::-1]


def downsample_smooth(path, gap, T=SE2State.T, cfg=VehicleConfig(), move_distance=SE2State.move_distance):
    if not path:
        print('no path ')
        return []
    ds_path = path[::gap]
    if len(ds_path) < 3:
        return ds_path
    else:
        for i in range(1, len(ds_path)-1):
            v_1 = (ds_path[i].x-ds_path[i-1].x)/T*m.cos(ds_path[i-1].heading) + \
                (ds_path[i].y-ds_path[i-1].y)/T*m.sin(ds_path[i-1].heading)
            v_2 = (ds_path[i+1].x-ds_path[i].x)/T*m.cos(ds_path[i].heading) + \
                (ds_path[i+1].y-ds_path[i].y)/T*m.sin(ds_path[i].heading)
            v = (v_1 + v_2)/2
            ds_path[i].v = v
        for i in range(len(ds_path)-1):
            ds_path[i].a += (ds_path[i+1].v - ds_path[i].v)/T
            diff_theta = ds_path[i+1].heading-ds_path[i].heading
            direction = 1
            if ds_path[i].v < 0:
                direction = -1
            steer = np.clip(m.atan(diff_theta*cfg.wheel_base/move_distance*direction),
                            -cfg.max_front_wheel_angle, cfg.max_front_wheel_angle)
            ds_path[i].steer = steer
        ds_path[-1] = path[-1]
        return ds_path


class SearchConfig:
    def __init__(self) -> None:
        self.max_iteration = 1900000
        self.max_heading_index_error = 2
        self.penalty_turn = 1.2
        self.penalty_change_gear = 1.6


def a_star_search(start: SE2State, goal: SE2State, grid_map, obstacles, search_cfg: SearchConfig = SearchConfig()):
    print('start :',start.x,start.y,start.heading)
    print('goal  :',goal.x,goal.y,goal.heading)
    q = PriorityQueue()
    close_list = []
    max_it = search_cfg.max_iteration
    max_heading_index_error = search_cfg.max_heading_index_error
    penalty_turn = search_cfg.penalty_turn
    penalty_change_gear = search_cfg.penalty_change_gear
    start.cost_to_hear = start.cost_to_state(start)
    start.cost_to_goal = start.cost_to_state(goal)
    q.put((start.cost(), start))
    it = 0
    map = grid_map.generate_map()
    print('searching...')
    while((not q.empty()) and (it < max_it)):
        pop = q.get()
        state_current = pop[1]
        it += 1
        map[state_current.x_index][state_current.y_index][state_current.heading_index] = state_current
        map[state_current.x_index][state_current.y_index][state_current.heading_index].visited = True
        close_list += [state_current]
        if(state_current.x_index == goal.x_index and
           state_current.y_index == goal.y_index and
           m.fabs(state_current.heading_index-goal.heading_index) < max_heading_index_error):
            print("goal reach!")
            print('iteration: ',it)
            return back_track(close_list, map)
        for next_state in state_current.get_next_states():
            if(map[next_state.x_index][next_state.y_index][next_state.heading_index].visited):
                continue
            elif(collsion(next_state, obstacles, max_x=grid_map.world_width, max_y=grid_map.world_height)):
                # print('collision ',next_state.direction_index)
                map[next_state.x_index][next_state.y_index][next_state.heading_index].visited = True
                continue
            else:
                next_state.cost_to_goal = next_state.cost_to_state(goal)
                if(next_state.direction_index == state_current.direction_index):
                    next_state.cost_to_hear = state_current.cost_to_hear + \
                        state_current.cost_to_state(next_state)
                elif((next_state.direction_index > 3 and state_current.direction_index <= 3) or
                     (next_state.direction_index <= 3 and state_current.direction_index > 3)):
                    next_state.cost_to_hear = state_current.cost_to_hear + \
                        state_current.cost_to_state(next_state)*penalty_turn
                else:
                    next_state.cost_to_hear = state_current.cost_to_hear + \
                        state_current.cost_to_state(
                            next_state)*penalty_change_gear
                next_state.parent = [
                    state_current.x_index, state_current.y_index, state_current.heading_index]
                q.put((next_state.cost(), next_state))
    print('search failed')
