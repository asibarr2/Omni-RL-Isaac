# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.utils.nucleus import find_nucleus_server
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.contact_sensor import _contact_sensor
from pxr import UsdGeom
import asyncio
import weakref
import omni.physx as _physx
import omni.ui as ui
import carb
import numpy as np
import omni


class CarterContact(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        from omni.isaac.core import World
        from omni.isaac.jetbot import Carter
        from omni.isaac.core.objects import VisualCuboid

        world = self.get_world()
        world.scene.add_default_ground_plane()
        #self.carter = self._world.scene.add(Carter(prim_path="/carter", name="my_carter", position=np.array([0, 0.0, 2.0]), orientation=np.array([1.0, 0.0, 0.0, 0.0]),))
        # add carter robot
        result, nucleus_server = find_nucleus_server()
        if result is False:
            # Use carb to log warnings, errors and infos in your application (shown on terminal)
            carb.log_error("Could not find nucleus server with /Isaac folder")
        asset_path = nucleus_server + "/Isaac/Robots/Carter/carter_v1.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/Fancy_Robot")
        carter_robot = world.scene.add(Robot(prim_path="/World/Fancy_Robot", name="fancy_robot"))

        # add walls
        self.add_walls(350,250)

        # Add collision properties between carter and walls
        import omni
        from pxr import Usd, UsdGeom, Gf, UsdPhysics
        from omni.physx.scripts import utils
        stage = omni.usd.get_context().get_stage()
        # Traverse all prims in the stage starting at this path
        curr_prim = stage.GetPrimAtPath("/")
        for prim in Usd.PrimRange(curr_prim):
            # only process shapes and meshes
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

        #collisionAPI = UsdPhysics.CollisionAPI.Apply(cubePrim)  # Add a Physics Collider, to be tested?

        # Add contact sensor to carter
        self._cs = _contact_sensor.acquire_contact_sensor_interface()
        #self._timeline = omni.timeline.get_timeline_interface()
        self.sub = _physx.get_physx_interface().subscribe_physics_step_events(self._on_update)

        props = _contact_sensor.SensorProperties()
        props.radius = -50 # (12) Sensor radius. Negative values indicate it’s a full body sensor. (float)
        props.minThreshold = 0 # Minimum force that the sensor can read. Forces below this value will not trigger a reading. (float)
        props.maxThreshold = 1000000000000 # Maximum force that the sensor can register. Forces above this value will be clamped. (float)
        props.sensorPeriod = 1 / 10.0 # (1/100.0) Sensor reading period in seconds. zero means sync with simulation timestep (float)
        props.position = carb.Float3(150, 0, 0) # Offset sensor 40cm in X direction from rigid body center

        self._sensor_handle = self._cs.add_sensor_on_body("/World/Fancy_Robot/chassis_link", props)

        # for multiple contact sensors:
        #self._sensor_handle = [0,0,0,0]
        #self._sensor_handle[0] = self._cs.add_sensor_on_body("/World/Fancy_Robot/left_wheel_link", props)
        #self._sensor_handle[1] = self._cs.add_sensor_on_body("/World/Fancy_Robot/right_wheel_link", props)
        #self._sensor_handle[2] = self._cs.add_sensor_on_body("/World/Fancy_Robot/chassis_link", props)
        #self._sensor_handle[3] = self._cs.add_sensor_on_body("/World/Fancy_Robot/chassis_link/com_link", props)

        # add lidar:
        from omni.isaac.range_sensor import _range_sensor # Imports the python bindings to interact with lidar sensor
        self.timeline = omni.timeline.get_timeline_interface()               # Used to interact with simulation
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface() # Used to interact with the LIDAR
        omni.kit.commands.execute('AddPhysicsSceneCommand',stage = stage, path='/World/PhysicsScene')
        self.lidarPath = "/LidarName"
        self.robotPath = "/World/Fancy_Robot/chassis_link/carter_lidar"
        result, prim = omni.kit.commands.execute(
                    "RangeSensorCreateLidar",
                    path=self.lidarPath,
                    parent=self.robotPath,
                    min_range=0.4,
                    max_range=100.0,
                    draw_points=False,
                    draw_lines=True,
                    horizontal_fov=360.0,
                    vertical_fov=30.0,
                    horizontal_resolution=0.4,
                    vertical_resolution=4.0,
                    rotation_rate=0.0,
                    high_lod=False,
                    yaw_offset=0.0,
                    enable_semantics=False
                )

        return

    def add_walls(self,distance,height):
        # Add walls to scene:
        from omni.isaac.core.objects import VisualCuboid
        self._world.scene.add(VisualCuboid(prim_path="/walls/wall_1", name="wall_1",position=np.array([distance,0,5]),size=np.array([10,2*distance,height]),color=np.array([1.0,1.0,1.0])))
        self._world.scene.add(VisualCuboid(prim_path="/walls/wall_2", name="wall_2",position=np.array([-distance,0,5]),size=np.array([10,2*distance,height]),color=np.array([1.0,1.0,1.0])))
        self._world.scene.add(VisualCuboid(prim_path="/walls/wall_3", name="wall_3",position=np.array([0,distance,5]),size=np.array([2*distance,10,height]),color=np.array([1.0,1.0,1.0])))
        self._world.scene.add(VisualCuboid(prim_path="/walls/wall_4", name="wall_4",position=np.array([0,-distance,5]),size=np.array([2*distance,10,height]),color=np.array([1.0,1.0,1.0])))
        return

    async def get_lidar_param(self):                                    # Function to retrieve data from the LIDAR
        await omni.kit.app.get_app().next_update_async()            # wait one frame for data
        #timeline = omni.timeline.get_timeline_interface() 
        #self.timeline.pause()                                            # Pause the simulation to populate the LIDAR's depth buffers
        #depth = self.lidarInterface.get_linear_depth_data(self.robotPath+self.lidarPath)
        #zenith = self.lidarInterface.get_zenith_data(self.robotPath+self.lidarPath)
        #azimuth = self.lidarInterface.get_azimuth_data(self.robotPath+self.lidarPath)
        pointcloud = self.lidarInterface.get_point_cloud_data(self.robotPath+self.lidarPath)
        print("pointcloud:", pointcloud)
        #print("depth", depth)                                      
        #print("zenith", zenith)
        #print("azimuth", azimuth)

    async def setup_post_load(self):
        self._world = self.get_world()
        self._carter = self._world.scene.get_object("fancy_robot")
        self._carter_articulation_controller = self._carter.get_articulation_controller()
        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        return

    def send_robot_actions(self, step_size):
        self._carter_articulation_controller.apply_action(ArticulationAction(joint_positions=None,
                                                                            joint_efforts=None,
                                                                            joint_velocities=5 * np.random.rand(4,)))
        return

    def _on_update(self, dt):
        # Read lidar data:
        #timeline = omni.timeline.get_timeline_interface() 
        #self.timeline.play()                                                 # Start the Simulation
        asyncio.ensure_future(self.get_lidar_param())                        # Only ask for data after sweep is complete
        
        # For multiple contact sensors:
        #for i, sensor in enumerate(self._sensor_handle):
        #    reading = self._cs.get_sensor_readings(sensor)
        #    print(reading)
        reading = self._cs.get_sensor_readings(self._sensor_handle) # returns (timestamp, force_value, inContact)
        if len(reading) > 0:
            print(reading)
            if reading[0][2] == True:
                print("Collision occurred")
        return
    


