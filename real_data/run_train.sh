#!/bin/bash
set -e
SECONDS=0


history_num_list=(3)
bad_bound_list=(1)

seed_list=({1..30})

feature_name_list=(
  # Single feature groups
  "MI_dim17"
  "PP_dim15"
  "TS_dim34"

  # Two-group combinations
  "MI_PP_dim32"
  "MI_TS_dim51"
  "PP_TS_dim49"

  # Three-group combination
  "MI_PP_TS_dim66"
)

lr_list=(0.001 0.0005 0.00075)
bs_list=(32 64 128 256)
dropout_p_list=(0)
hidden_num_list=(3)
hidden_dim_list=(16 32)
weight_decay_list=(0.0001)


commands=()

for seed in "${seed_list[@]}"; do
    for lr in "${lr_list[@]}"; do
        for bs in "${bs_list[@]}"; do
            for dropout_p in "${dropout_p_list[@]}"; do
                for hidden_num in "${hidden_num_list[@]}"; do
                    for hidden_dim in "${hidden_dim_list[@]}"; do
                        for history_num in "${history_num_list[@]}"; do
                            for bad_bound in "${bad_bound_list[@]}"; do
                                for feature_name in "${feature_name_list[@]}"; do
                                    for weight_decay in "${weight_decay_list[@]}"; do
                                    # Generate commands for training
                                        commands+=("python main_train_realdata.py --sim_id ${seed} --lr ${lr} --bs ${bs} --dropout_p ${dropout_p} --hidden_num ${hidden_num} --hidden_dim ${hidden_dim} --history_num ${history_num} --bad_player_bound ${bad_bound} --feature_name ${feature_name} --weight_decay ${weight_decay}")
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
    echo "Interrupted. Stopping running training jobs..." >&2
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
