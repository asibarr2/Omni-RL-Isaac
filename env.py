# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import gym
from gym import spaces
import numpy as np
import math
import asyncio

class JetBotEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=1500,
        seed=0,
        headless=True,
        goal_threshold=5,
        wall_threshold=5,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._goal_threshold = goal_threshold
        self._wall_threshold = wall_threshold
        self._steps_after_reset = int(rendering_dt / physics_dt)
        self.high = np.array([np.inf]*9).astype(np.float32)
        print(self.high)

        from omni.isaac.core import World
        from omni.isaac.jetbot import Carter        
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.core.objects import DynamicCuboid
        from omni.isaac.contact_sensor import _contact_sensor
        import omni
        import omni.physx as _physx
        import omni.ui as ui
        import carb

        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=0.01)
        self._my_world.scene.add_default_ground_plane()
        
        self._cs = _contact_sensor.acquire_contact_sensor_interface()
        self.sub = _physx.get_physx_interface().subscribe_physics_step_events(self._on_update)
        
        props = _contact_sensor.SensorProperties()
        props.radius = -50 # (12) Sensor radius. Negative values indicate it’s a full body sensor. (float)
        props.minThreshold = 0 # Minimum force that the sensor can read. Forces below this value will not trigger a reading. (float)
        props.maxThreshold = 1000000000000 # Maximum force that the sensor can register. Forces above this value will be clamped. (float)
        props.sensorPeriod = 1 / 10.0 # (1/100.0) Sensor reading period in seconds. zero means sync with simulation timestep (float)
        props.position = carb.Float3(150, 0, 0) # Offset sensor 40cm in X direction from rigid body center

        self._sensor_handle = self._cs.add_sensor_on_body("/World/carter/chassis_link", props)

        
        self.timeline = omni.timeline.get_timeline_interface()               # Used to interact with simulation
        self.timeline.play()
        
        from pxr import Usd, UsdGeom, Gf, UsdPhysics
        from omni.physx.scripts import utils
        stage = omni.usd.get_context().get_stage()
        curr_prim = stage.GetPrimAtPath("/")
        for prim in Usd.PrimRange(curr_prim):
            #only process shapes and meshes
            if (
                prim.IsA(UsdGeom.Cylinder)
                or prim.IsA(UsdGeom.Capsule)
                or prim.IsA(UsdGeom.Cone)
                or prim.IsA(UsdGeom.Sphere)
                or prim.IsA(UsdGeom.Cube)
            ):
            
                # use a ConvexHull for regular prims
                utils.setCollider(prim, approximationShape="convexHull")
        pass
        
        from omni.isaac.occupancy_map import _occupancy_map

        physx = omni.physx.acquire_physx_interface()
        stage_id = omni.usd.get_context().get_stage_id()


        generator = _occupancy_map.Generator(physx, stage_id)
        generator.update_settings(5, 4, 5, 6)
        # Set location to map from and the min and max bounds to map to
        generator.set_transform((0, 0, 0), (-1200, -1200, 38), (1200, 1200, 0))
        generator.generate()
        # Get locations of the occupied cells in the stage
        points = generator.get_occupied_positions()
        # Get computed 2d occupancy buffer
        buffer = generator.get_buffer()
        # Get dimensions for 2d buffer
        dims = generator.get_dimensions()

        print(points)
        
        self.carter = self._my_world.scene.add(
            # Carter Class instead of Jetbot Class
            Carter(
                prim_path="/World/carter",
                name="my_carter",
                position=np.array([0, 0.0, 2.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )

        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=np.array([100, 130, 20.5]),
                size=np.array([20, 20, 70]),
                color=np.array([1.0, 0, 0]),
            )
        )
        
        self.south_east = self._my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/se_wall",
                name="se_wall",
                position=np.array([-100, 950, 0.0]),
                mass=300,
                size=np.array([2000, 50.15, 200.15]),
                color=np.array([0.0, 0.0, 1.0]),
            )
        )

        self.south_west = self._my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/sw_wall",
                name="sw_wall",
                position=np.array([900, -32, 0.0]),
                mass=300,
                size=np.array([50.15, 2000, 200.15]),
                color=np.array([0.0, 0.0, 1.0]),
            ))
        
        self.north_west = self._my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/nw_wall",
                name="nw_wall",
                position=np.array([-100, -1050, 0.0]),
                mass=300,
                size=np.array([2000, 50.15, 200.15]),
                color=np.array([0.0, 0.0, 1.0]),
            )
        )
        
        self.north_east = self._my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/ne_wall",
                name="ne_wall",
                position=np.array([-1140, -50, 0.0]),
                mass=300,
                size=np.array([50.15, 2000, 200.15]),
                color=np.array([0.0, 0.0, 1.0]),
            )
        )
        
        

        self.seed(seed)
        self.sd_helper = None
        self.viewport_window = None
        self._set_camera()
        self._set_wall()
        self.timeline.play()
        asyncio.ensure_future(self.get_lidar_obs())
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=0.0, high=10, shape=(2,), dtype=np.float32)
        #self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        self.test = {
            "rgb": spaces.Box(low=0.0, high=255.0, shape=(128, 128, 3), dtype=np.uint8),
            "vector": spaces.Box(-self.high, self.high)  
        }
        # observation space for the lidar
        #self.observation_space = spaces.Box(-self.high, self.high)
        self.observation_space = spaces.Dict(self.test)
        return

    def get_dt(self):
        return self._dt

    def step(self, action):
        previous_jetbot_position, _ = self.carter.get_world_pose()
        for i in range(self._skip_frame):
            from omni.isaac.core.utils.types import ArticulationAction

            self.carter.apply_wheel_actions(ArticulationAction(joint_velocities=action * 10.0))
            self._my_world.step(render=False)
        observations = self.get_observations(), self.get_lidar_obs()
        info = {}
        done = False
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True
        goal_world_position, _ = self.goal.get_world_pose()
        current_jetbot_position, _ = self.carter.get_world_pose()
        se_wall_position, _ = self.south_east.get_world_pose()
        sw_wall_position, _ = self.south_west.get_world_pose()
        ne_wall_position, _ = self.north_east.get_world_pose()
        nw_wall_position, _ = self.north_west.get_world_pose()

        # (+) Reward is given for reaching the cube
        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_jetbot_position)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_jetbot_position)
        reward = previous_dist_to_goal - current_dist_to_goal
        if current_dist_to_goal <= self._goal_threshold:
            reward += 200
            done = True
        
        # If Robot is near wall, it will deduct 100 from reward
        #previous_dist_to_se_wall = se_wall_position - previous_jetbot_position)
        current_dist_to_sw_wall = np.linalg.norm(sw_wall_position - current_jetbot_position)
        current_dist_to_se_wall = np.linalg.norm(se_wall_position - current_jetbot_position)
        current_dist_to_ne_wall = np.linalg.norm(ne_wall_position - current_jetbot_position)
        current_dist_to_nw_wall = np.linalg.norm(nw_wall_position - current_jetbot_position)


        if (current_dist_to_sw_wall <= self._wall_threshold) or (current_dist_to_se_wall <= self._wall_threshold) or (current_dist_to_ne_wall <= self._wall_threshold) or (current_dist_to_nw_wall <= self._wall_threshold):
            reward -= 100
            done = True
                
        return observations, reward, done, info

    def reset(self):
        self._my_world.reset()
        # randomize goal location in circle around robot
        alpha = 2 * math.pi * np.random.rand()
        r = 100 * math.sqrt(np.random.rand()) + 150 # radius = randomdist + minimumDistance
        self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, 2.5]))
        observations = self.get_observations()
        return observations

    def get_observations(self):
        self._my_world.render()
        # wait_for_sensor_data is recommended when capturing multiple sensors, in this case we can set it to zero as we only need RGB
        gt = self.sd_helper.get_groundtruth(
            ["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
        )
        #print(type(gt["rgb"][:, :, :3]))

        return gt["rgb"][:, :, :3]
    
    async def get_lidar_obs(self):
        from omni.isaac.range_sensor._range_sensor import acquire_lidar_sensor_interface
        from omni.isaac.range_sensor import _range_sensor
        import omni
        await asyncio.sleep(1.0)        
        self.timeline.pause()
        lidarInterface = acquire_lidar_sensor_interface()
        lidarPath = "/carter/chassis_link/carter_lidar"
        if lidarInterface.is_lidar_sensor("/World"+lidarPath):
            print("Lidar sensor is valid")
        depth = lidarInterface.get_linear_depth_data("/World"+lidarPath) # print the data

        scan = np.array([depth[0], depth[112], depth[225], depth[337], depth[450], depth[562], depth[675], depth[787], depth[899]])
        print(scan.shape)
        
        
    
    """async def _get_lidar_observations(self):
        self._my_world.render()
        from omni.isaac.range_sensor._range_sensor import acquire_lidar_sensor_interface
        from omni.isaac.range_sensor import _range_sensor
        import omni
        await omni.kit.app.get_app().next_update_async()
        
        lidarInterface = acquire_lidar_sensor_interface()
        lidarPath = "/carter/chassis_link/carter_lidar"
        labels = lidarInterface.get_semantic_data("/World"+lidarPath)
        return labels
    """    

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def _set_camera(self):
        import omni.kit
        from omni.isaac.synthetic_utils import SyntheticDataHelper

        camera_path = "/World/carter/chassis_link/camera_mount/carter_camera_first_person"
        if self.headless:
            viewport_handle = omni.kit.viewport.get_viewport_interface()
            viewport_handle.get_viewport_window().set_active_camera(str(camera_path))
            viewport_window = viewport_handle.get_viewport_window()
            self.viewport_window = viewport_window
            viewport_window.set_texture_resolution(128, 128)
        else:
            viewport_handle = omni.kit.viewport.get_viewport_interface().create_instance()
            new_viewport_name = omni.kit.viewport.get_viewport_interface().get_viewport_window_name(viewport_handle)
            viewport_window = omni.kit.viewport.get_viewport_interface().get_viewport_window(viewport_handle)
            viewport_window.set_active_camera(camera_path)
            viewport_window.set_texture_resolution(128, 128)
            viewport_window.set_window_pos(1000, 400)
            viewport_window.set_window_size(420, 420)
            self.viewport_window = viewport_window
        self.sd_helper = SyntheticDataHelper()
        self.sd_helper.initialize(sensor_names=["rgb"], viewport=self.viewport_window)
        self._my_world.render()
        self.sd_helper.get_groundtruth(["rgb"], self.viewport_window)
        return
    
    
    def _set_wall(self):
        
        # Call the walls to be added to the scene
        self._se = self._my_world.scene.get_object("se_wall")
        self._sw = self._my_world.scene.get_object("sw_wall")
        self._nw = self._my_world.scene.get_object("nw_wall")
        self._ne = self._my_world.scene.get_object("ne_wall")
        # Render the walls so that they appear in the scene
        self._my_world.render()
        return
    
   
    # Objective is to make lidar an observation    
    """async def _set_lidar(self):
        from omni.isaac.range_sensor._range_sensor import acquire_lidar_sensor_interface
        from omni.isaac.range_sensor import _range_sensor
        import omni
        
        await omni.kit.app.get_app().next_update_async()
        lidarInterface = acquire_lidar_sensor_interface()
        lidarPath = "/World/carter/chassis_link/carter_lidar"
        
        pointcloud = lidarInterface.get_point_cloud_data(lidarPath)
        #print("pointcloud:", pointcloud)
        
        #if lidarInterface.is_lidar_sensor("/World"+lidarPath):
        #    print("Lidar sensor is valid")
        #depth = lidarInterface.get_linear_depth_data("/World"+lidarPath)
        #labels = lidarInterface.get_semantic_data("/World"+lidarPath)
    """
        
    def _on_update(self, dt):
        
        
        reading = self._cs.get_sensor_readings(self._sensor_handle)
        if len(reading) > 0:
            print(reading)
            if reading[0][2] == True:
                print("Collision occured")
            else:
                print("No collision thus far")
        return
        