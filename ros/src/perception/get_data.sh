#!/bin/sh

rsync -avp -e ssh --progress $ROBOT:RoboCar/data/ /home/wolfgang/RoboCar/data.$ROBOT/
