#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

python server.py --fraction-fit 0.5 --min-available-clients 2 --num-rounds 100 --strategy FedAvgM &
sleep 3  # Sleep for 3s to give the server enough time to start

# HOW TO USE:
# To change the number of communication rounds, change --num-rounds on line 5
# To change the number of clients (can be 2 to 15), change the for loop on line 12 and the --num-clients= on line 14

for i in `seq 0 14`; do
    echo "Starting client $i"
    python client.py --partition=1 --num-clients=15 --data-path data/learn_clinics/clinic_${i} --eval-dataset test&
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait