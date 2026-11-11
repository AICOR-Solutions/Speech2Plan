from __future__ import annotations

import threading
from dataclasses import dataclass, field

from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.world import World

from speech2plan.world_source import WorldSource

# %% the live world


@dataclass
class FetchedWorld(WorldSource):
    """Fetches a world snapshot from whichever node runs semdt's FetchWorldServer."""

    node: Node = field(kw_only=True)
    """A node no executor spins yet; it is spun while fetching."""

    service_suffix: str = field(kw_only=True, default="fetch_world")
    timeout_seconds: float = field(kw_only=True, default=60.0)
    """How long to wait for the snapshot: the server serializes the whole world
    first, which takes over ten seconds for the IAI office with a TIAGo in it."""

    def is_available(self) -> bool:
        return any(
            name.endswith(self.service_suffix)
            and service_type == ["std_srvs/srv/Trigger"]
            for name, service_type in self.node.get_service_names_and_types()
        )

    def load(self) -> World:
        """Fetch while `node` spins on another thread, which the blocking service
        call needs to receive the response."""
        executor = SingleThreadedExecutor()
        executor.add_node(self.node)
        spinning = threading.Thread(target=executor.spin, daemon=True)
        spinning.start()
        try:
            return fetch_world_from_service(
                self.node,
                service_suffix=self.service_suffix,
                timeout_seconds=self.timeout_seconds,
            )
        finally:
            executor.shutdown()
            spinning.join()
