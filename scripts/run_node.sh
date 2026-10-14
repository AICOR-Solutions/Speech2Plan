#!/usr/bin/env bash
# Run the node with the ROS environment it needs.
#
# Sources ROS and the colcon workspace that holds pyhri and the skill
# interfaces, then runs the node. Set SPEECH2PLAN_WORKSPACE to use another
# workspace, ROS_DISTRO or SPEECH2PLAN_ROS_DISTRO to use another ROS.
set -e

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${SPEECH2PLAN_WORKSPACE:-$(dirname "$repository")}"

if [ -z "${AMENT_PREFIX_PATH:-}" ]; then
  source "/opt/ros/${ROS_DISTRO:-${SPEECH2PLAN_ROS_DISTRO:-jazzy}}/setup.bash"
fi
if [ -f "$workspace/install/setup.bash" ]; then
  source "$workspace/install/setup.bash"
else
  echo "no colcon workspace at $workspace/install, pyhri will be missing" >&2
fi

export PYTHONPATH="$repository/src:${PYTHONPATH:-}"
exec python3 -u -m speech2plan.ros.intent_node "$@"
