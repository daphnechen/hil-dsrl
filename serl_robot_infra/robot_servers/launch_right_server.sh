# Source the setup.bash file for the second ROS workspace
source /home/daphne/ros_noetic_serl/devel/setup.zsh

# Change the ROS master URI to a different port
export ROS_MASTER_URI=http://localhost:11311

# Run the second instance of franka_server.py in the background
python franka_server.py \
    --robot_ip=172.16.0.2 \
    --gripper_type=Franka \
    --reset_joint_target=0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741 \
    --flask_url=127.0.0.1 \
    --ros_port=11311
