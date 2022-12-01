#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  REMOTE=`avahi-resolve-host-name raspberrypi.local -4 | awk '{print $2}'`
elif [[ "$OSTYPE" == "darwin"* ]]; then
  REMOTE=raspberrypi.local
else
  print "not supported os"
  exit 1
fi

echo PI addrees is "${REMOTE}"
REMOTE_DIR=/home/pi/puppersim_deploy

ssh pi@${REMOTE}  mkdir -p ${REMOTE_DIR}
rsync -avz ${PWD} pi@${REMOTE}:${REMOTE_DIR}

if [ -z "$1" ] ; then
  ssh -t pi@${REMOTE} "cd ${REMOTE_DIR} ; bash --login"
else
  ssh -t pi@${REMOTE} "cd ${REMOTE_DIR} ; $@"
fi
