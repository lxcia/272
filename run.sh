#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

python server.py --fraction-fit 0.5 --min-available-clients 2 --num-rounds 25 &
sleep 15  # Sleep for 3s to give the server enough time to start

# to change com rounds, change the --num-rounds on line 5
# to change client nums, change the for loop on line 11 and the --num-clients on line 13

for i in `seq 0 2`; do
    echo "Starting client $i"
     python client.py --partition=1 --num-clients=3 --data-path data/clinic_${i} --eval-dataset test&
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait