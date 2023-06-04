#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

python server.py --fraction-fit 0.5 --min-available-clients 2 --num-rounds 2&
sleep 15  # Sleep for 3s to give the server enough time to start

for i in `seq 0 1`; do
    echo "Starting client $i"
     python client.py --partition=1 --num-clients=2 --data-path data/clinic_${i} --eval-dataset test&
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait