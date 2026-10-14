from __future__ import annotations

from dataclasses import dataclass, field

from rclpy.node import Node
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.world import World

from speech2plan.world_source import WorldSource

# %% the live world


@dataclass
class FetchedWorld(WorldSource):
    """Fetches a world snapshot from whichever node runs semdt's FetchWorldServer."""

    node: Node = field(kw_only=True)
    service_suffix: str = field(kw_only=True, default="fetch_world")
    timeout_seconds: float = field(kw_only=True, default=10.0)

    def is_available(self) -> bool:
        return any(
            name.endswith(self.service_suffix)
            and service_type == ["std_srvs/srv/Trigger"]
            for name, service_type in self.node.get_service_names_and_types()
        )

    def load(self) -> World:
        return fetch_world_from_service(
            self.node,
            service_suffix=self.service_suffix,
            timeout_seconds=self.timeout_seconds,
        )
