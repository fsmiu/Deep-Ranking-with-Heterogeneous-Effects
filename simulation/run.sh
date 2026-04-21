#!/bin/bash
set -e
SECONDS=0

seed_list=({1..300})
n_player=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
lr_list=(0.01)
weight_decay_list=(0.00001)
bs_list=(512)
dropout_p_list=(0)
hidden_num_list=("no")
hidden_dim_list=(128)
x_function_type_list=("dynamic_complex")
pow_logn_list=(0.05 0.1 0.15 0.2 0.25)


# modes=("zeroshot")

commands=()

for seed in "${seed_list[@]}"; do
    for lr in "${lr_list[@]}"; do
        for bs in "${bs_list[@]}"; do
            for dropout_p in "${dropout_p_list[@]}"; do
                for hidden_num in "${hidden_num_list[@]}"; do
                    for hidden_dim in "${hidden_dim_list[@]}"; do
                        for n in "${n_player[@]}"; do
                            for weight_decay in "${weight_decay_list[@]}"; do
                                for x_function_type in "${x_function_type_list[@]}"; do
                                    for pow_logn in "${pow_logn_list[@]}"; do
                                # Generate commands for training
                                        commands+=("python simulation.py  --sim_id ${seed} --lr ${lr} --bs ${bs} --dropout_p ${dropout_p} --hidden_dim ${hidden_dim} --n ${n} --weight_decay ${weight_decay} --x_function_type ${x_function_type} --power_logn ${pow_logn}")
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done


# Number of concurrent processes
CONCURRENT=1
# Total number of commands
TOTAL_COMMANDS=${#commands[@]}
# Index to track the next command to run
next_index=0
# Arrays to store running process PIDs and their command indexes.
running_pids=()
running_indices=()
running_status_files=()
running_child_pid_files=()
status_dir=$(mktemp -d)

cleanup() {
    trap - EXIT INT TERM
    for pid in "${running_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid_file in "${running_child_pid_files[@]}"; do
        if [ -f "$pid_file" ]; then
            child_pid=$(< "$pid_file")
            kill "$child_pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    rm -rf "$status_dir"
}

interrupt() {
    echo
    echo "Interrupted. Stopping running simulation jobs..." >&2
    cleanup
    exit 130
}

trap cleanup EXIT
trap interrupt INT TERM

# Function to start a new command if there are commands left
start_command() {
    if [ "$next_index" -lt "$TOTAL_COMMANDS" ]; then
        cmd="${commands[$next_index]}"
        cmd_index=$next_index
        status_file="${status_dir}/job_${cmd_index}.status"
        child_pid_file="${status_dir}/job_${cmd_index}.pid"
        echo "Starting: $cmd (Index: $next_index)"
        (
            set +e
            child_pid=""
            stop_child() {
                if [ -n "$child_pid" ]; then
                    kill -INT "$child_pid" 2>/dev/null || true
                    wait "$child_pid" 2>/dev/null || true
                fi
                exit 130
            }
            trap stop_child INT TERM

            bash -c "$cmd" &
            child_pid=$!
            printf "%s\n" "$child_pid" > "$child_pid_file"
            wait "$child_pid"
            status=$?
            trap - INT TERM
            printf "%s\n" "$status" > "${status_file}.tmp"
            mv "${status_file}.tmp" "$status_file"
        ) &
        pid=$!
        running_pids+=("$pid")
        running_indices+=("$cmd_index")
        running_status_files+=("$status_file")
        running_child_pid_files+=("$child_pid_file")
        ((++next_index))
    fi
}

# Start initial batch of CONCURRENT commands
for ((i=0; i<CONCURRENT && i<TOTAL_COMMANDS; i++)); do
    start_command
done

# Monitor and maintain CONCURRENT running processes.
failed=0
while [ "${#running_pids[@]}" -gt 0 ]; do
    for i in "${!running_pids[@]}"; do
        pid="${running_pids[$i]}"
        status_file="${running_status_files[$i]}"
        if [ -f "$status_file" ]; then
            cmd_index="${running_indices[$i]}"
            status=$(< "$status_file")
            wait "$pid" || true
            if [ "$status" -eq 0 ]; then
                echo "Completed: ${commands[$cmd_index]} (PID: $pid)"
            else
                echo "Failed with exit code $status: ${commands[$cmd_index]} (PID: $pid)" >&2
                failed=$status
            fi

            unset 'running_pids[i]'
            unset 'running_indices[i]'
            unset 'running_status_files[i]'
            unset 'running_child_pid_files[i]'
            running_pids=("${running_pids[@]}")
            running_indices=("${running_indices[@]}")
            running_status_files=("${running_status_files[@]}")
            running_child_pid_files=("${running_child_pid_files[@]}")

            if [ "$failed" -ne 0 ]; then
                cleanup
                exit "$failed"
            fi

            start_command
            break
        fi
    done
    # Small sleep to avoid excessive CPU usage
    sleep 0.1
done

total=$SECONDS
printf "Total elapsed: %02d:%02d:%02d\n" $((total/3600)) $(((total%3600)/60)) $((total%60))

echo "All commands completed"
