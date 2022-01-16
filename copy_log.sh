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
REMOTE_DIR=/home/pi/puppersim_deploy/env_log.txt
rsync -avz  pi@${REMOTE}:${REMOTE_DIR} env_log_real.txt

