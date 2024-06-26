#!/usr/bin/env bash
echo "PYTHONPATH is ${PYTHONPATH}"

algorithms_dir="${PWD}/jobs"

job=$1
poc=$2

if [ "${poc}" ]
then
  if [ "${poc}" != "--poc" ]; then
    echo "${poc} not supported to run POC mode. Use --poc"
  fi
  echo "Submit poc job ${job}"
  workspace="/tmp/nvflare/poc/example_project/prod_00"
  admin_username="admin@nvidia.com"
else
  echo "Submit test job ${job}"
  workspace="${PWD}/workspaces/test_workspace"
  admin_username="rosa.llorente.alonso@alumnos.upm.es"
fi

# submit job
COMMAND="python3 ./submit_job.py --admin_dir=${workspace}/${admin_username} --username=${admin_username} --job=${algorithms_dir}/${job} ${poc}"
echo "Running: ${COMMAND}"
eval "${COMMAND}"