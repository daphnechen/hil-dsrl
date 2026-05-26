# Source the setup.bash file for the second ROS workspace
source /home/daphne/ros_noetic_serl/devel/setup.bash

# Add serl_robot_infra to PYTHONPATH
export PYTHONPATH=/home/daphne/Desktop/jax-hitl-hil-serl/serl_robot_infra:$PYTHONPATH

# Change the ROS master URI to a different port
export ROS_MASTER_URI=http://localhost:11511

sudo usermod -a -G dialout $USER

sudo chmod a+rw /dev/ttyUSB0
sudo chmod 777 /dev/ttyUSB0

# Run the second instance of franka_server.py in the background
python franka_server.py \
    --robot_ip=172.16.0.2 \
    --gripper_type=Robotiq \
    --reset_joint_target=0.0,-0.569,0.0,-2.810,0.0,3.037,0.741 \
    --flask_url=127.0.0.2 \
    --ros_port=11511
