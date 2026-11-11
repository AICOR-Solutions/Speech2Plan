cmake_minimum_required(VERSION 3.8)
project(speech2plan_msgs)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/AskForHelp.action"
  "action/CarryOutIntent.action"
  DEPENDENCIES hri_actions_msgs std_skills
)

ament_export_dependencies(rosidl_default_runtime)

ament_auto_package()
