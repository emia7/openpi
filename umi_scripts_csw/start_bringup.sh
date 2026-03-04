#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/noetic/setup.bash


source ~/catkin_ws/devel/setup.bash

source ~/catkin_dual/devel/setup.bash

roslaunch fastumi_dual_bringup bringup_10hz.launch