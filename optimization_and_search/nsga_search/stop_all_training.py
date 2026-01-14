from nsga2 import Population
from typing import List, Dict, Any
from search_space import Individual
from search_space import HeteroSearchSpace
import yaml
from remote_trainer import RemoteTrainer  
import logging
import time
from run_exp import load_hosts_from_file
import os
import argparse


# Configure logging to only show INFO:root messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
# Disable all other loggers except root
for name in ("paramiko", "paramiko.transport", "fabric", "invoke"):
    logging.getLogger(name).disabled = True

def main():
    parser = argparse.ArgumentParser(description="Stop/clear all remote training jobs across hosts.")
    parser.add_argument(
        "--hosts",
        type=str,
        default="../host_configs/internal_hosts.yaml",
        help="Path to YAML file listing remote hosts",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="xinting",
        help="SSH username",
    )
    parser.add_argument(
        "--key",
        dest="key_filename",
        type=str,
        default="/home/xinting/.ssh/id_rsa",
        help="Path to SSH private key file",
    )

    args = parser.parse_args()

    hosts = load_hosts_from_file(args.hosts)
    user = args.user
    key_filename = args.key_filename

    trainer = RemoteTrainer(hosts=hosts, user=user, key_filename=key_filename)
    trainer.clear_all_jobs()
    print("Cleared all training jobs on remote hosts.")


if __name__ == "__main__":
    main()



