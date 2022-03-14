import os
from omni.isaac.examples.base_sample import BaseSampleExtension
from omni.isaac.examples.user_examples import CarterContact


class CarterContactExtension(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="",
            submenu_name="",
            name="Carter Contact",
            title="Carter V1 Contact Sensor Testing",
            doc_link="https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_required_hello_world.html",
            overview="Contact Sensor example with Carter V1 robot.",
            file_path=os.path.abspath(__file__),
            sample=CarterContact(),
        )
        return
