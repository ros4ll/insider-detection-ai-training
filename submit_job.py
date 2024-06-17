import argparse
import json
import os
import uuid

from nvflare.fuel.hci.client.fl_admin_api_runner import FLAdminAPIRunner, api_command_wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--admin_dir", type=str, default="./admin/", help="Path to admin directory.")
    parser.add_argument("--username", type=str, default="rosa.llorente.alonso@alumnos.upm.es", help="Admin username.")
    parser.add_argument("--job", type=str, default="cifar10_fedavg", help="Path to job")
    parser.add_argument("--poc", action="store_true", help="Whether admin does not use SSL.")

    args = parser.parse_args()

    assert os.path.isdir(args.admin_dir), f"admin directory does not exist at {args.admin_dir}"

    # Initialize the runner
    runner = FLAdminAPIRunner(
        username=args.username,
        admin_dir=args.admin_dir,
        poc=args.poc,
        debug=False,
    )

    # Submit job
    api_command_wrapper(runner.api.submit_job(args.job))

    # finish
    runner.api.logout()


if __name__ == "__main__":
    main()