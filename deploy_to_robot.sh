#!/bin/bash

# REMOTE=localhost
REMOTE=`avahi-resolve-host-name raspberrypi.local -4 | awk '{print $2}'`
echo PI addrees is "${REMOTE}"
REMOTE_DIR=/home/pi/puppersim_deploy

ssh pi@${REMOTE}  mkdir -p ${REMOTE_DIR}
rsync -avz ${PWD} pi@${REMOTE}:${REMOTE_DIR}

if [ -z "$1" ] ; then
  ssh -t pi@${REMOTE} "cd ${REMOTE_DIR} ; bash --login"
else
  ssh -t pi@${REMOTE} "cd ${REMOTE_DIR} ; $@"
fi
