from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .nodes import (
    H3OptimizerRef2VAPromptPackage,
    H3OptimizerRef2VAPromptPackageGenerator,
    H3OptimizerSegmentSettings,
    H3OptimizerVideoOutput,
)
from . import routes as _routes


class H3PromptOptimizerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            H3OptimizerSegmentSettings,
            H3OptimizerRef2VAPromptPackageGenerator,
            H3OptimizerRef2VAPromptPackage,
            H3OptimizerVideoOutput,
        ]


async def comfy_entrypoint() -> H3PromptOptimizerExtension:
    return H3PromptOptimizerExtension()


NODE_CLASS_MAPPINGS = {
    "H3OptimizerSegmentSettingsCS": H3OptimizerSegmentSettings,
    "H3OptimizerRef2VAPromptPackageGeneratorCS": H3OptimizerRef2VAPromptPackageGenerator,
    "H3OptimizerRef2VAPromptPackageCS": H3OptimizerRef2VAPromptPackage,
    "H3OptimizerVideoOutputCS": H3OptimizerVideoOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3OptimizerSegmentSettingsCS": "H3 Optimizer Segment Settings",
    "H3OptimizerRef2VAPromptPackageGeneratorCS": "H3 Optimizer Ref2VA Prompt Package Generator",
    "H3OptimizerRef2VAPromptPackageCS": "H3 Optimizer Ref2VA Prompt Package",
    "H3OptimizerVideoOutputCS": "H3 Optimizer Video Output",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
