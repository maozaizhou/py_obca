import numpy as np
import math as m
import matplotlib.pyplot as plt
from pypoman import plot_polygon
from cProfile import label
import sys
from pip import main
sys.path.append('../')
import pyobca


def parallel_parking():
    # build map and obstacle
    map = pyobca.GridMap(20, 10)
    parking_obs1 = [
        [20, 2.3],
        [11.3, 2.3],
        [11.3, 0],
        [20, 0]
    ]

    parking_obs2 = [
        [6.0, 2.3],
        [0, 2.3],
        [0, 0],
        [6.0, 0]
    ]
    obstacles = [parking_obs1, parking_obs2]
    start = pyobca.SE2State(7.5, 4.0, 0.)
    goal = pyobca.SE2State(7.6, 1.25, 0.0)

    # coarse search
    search_path = pyobca.a_star_search(start, goal, map, obstacles)
    ds_path = pyobca.downsample_smooth(path=search_path, gap=1)
    if len(ds_path)<2:
        print('no enough path point')
        return 
    init_x = []
    init_y = []
    for state in ds_path:
        init_x += [state.x]
        init_y += [state.y]
    # obca optimization
    optimizer = pyobca.OBCAOptimizer()
    optimizer.initialize(ds_path, obstacles,
                         max_x=map.world_width, max_y=map.world_height)
    optimizer.build_model()
    optimizer.generate_constrain()
    optimizer.generate_variable()
    r = [[0.1, 0], [0, 0.1]]
    q = [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0., 0],
         [0, 0, 0, 0, 0],
         ]
    optimizer.generate_object(r, q)
    optimizer.solve()


    x_opt = optimizer.x_opt.elements()
    y_opt = optimizer.y_opt.elements()
    heading_opt = optimizer.theta_opt.elements()
    steer_opt = optimizer.steer_opt.elements()

    # visualization
    fig, ax = plt.subplots()
    ax.plot(x_opt, y_opt, 'go', ms=3, label='optimized path')
    ax.plot(init_x, init_y, 'ro', ms=3, label='coarse path')
    ax.set_xlim(0,20)
    plot_polygon(parking_obs1)
    plot_polygon(parking_obs2, color='r')
    ax.legend()
    result_state = np.array([x_opt, y_opt, heading_opt]).T
    for state in result_state:
        verts = pyobca.generate_vehicle_vertices(
            pyobca.SE2State(state[0], state[1], state[2]), base_link=True)
        plot_polygon(verts, fill=False, color='b')
    v_opt = optimizer.v_opt.elements()
    a_opt = optimizer.a_opt.elements()
    t = [optimizer.T*k for k in range(len(v_opt))]
    t_a = [optimizer.T*k for k in range(len(a_opt))]
    fig2, ax2 = plt.subplots(3)
    ax2[0].plot(t, v_opt, label='v-t')
    ax2[1].plot(t_a, a_opt, label='a-t')
    ax2[2].plot(t, steer_opt, label='steering-t')
    ax2[0].legend()
    ax2[1].legend()
    ax2[2].legend()
    plt.show()


if __name__ == "__main__":
    parallel_parking()
