#!/bin/bash

# Define the process names to search for
PROCESS_NAMES="roscore|roslaunch|rosmaster"

# Find the Process IDs (PIDs)
PIDS=$(pgrep -f "$PROCESS_NAMES")

if [ -z "$PIDS" ]; then
    echo "No ROS processes matching '$PROCESS_NAMES' were found."
else
    echo "Terminating the following PIDs: $PIDS"
    # Send a termination signal to the identified processes
    kill -9 $PIDS
    echo "Termination signal sent."
fi