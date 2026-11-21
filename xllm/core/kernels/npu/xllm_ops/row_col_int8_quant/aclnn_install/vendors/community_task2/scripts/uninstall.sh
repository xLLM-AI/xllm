#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

vendor_name=community_task2

curr_path=$(dirname "$0")
vendordir=$(realpath ${curr_path}/../../)
custom_delete=$vendordir/$vendor_name

log() {
    cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[ops_custom] [$cur_date] "$1
}

if [[ ! -d $custom_delete ]]; then
    log "[INFO] no need to delete ops $custom_delete files"
    exit 1
else
    log "[INFO] Starting to delete $vendor_name ..."
    chmod -R +w $custom_delete
    [ -d "$custom_delete" ] && rm -rf $custom_delete
    if [ $? -ne 0 ]; then
        log "[ERROR] Failed to delete $vendor_name."
        exit 1
    fi
fi

config_file=${vendordir}/config.ini
if [ ! -f "$config_file" ]; then
    log "[ERROR] File '$config_file' does not exist."
    exit 1
fi

line=$(grep -E '^\s*load_priority\s*=' "$config_file" | head -n1)
if [ -z "$line" ]; then
    log "[INFO] No 'load_priority=' found in '$config_file'."
    exit 0
fi

value=$(echo "$line" | sed -E 's/^[^=]*=\s*(.*)/\1/' | tr -d ' \t')
IFS=',' read -r -a items <<< "$value"
new_items=()
for item in "${items[@]}"; do
    if [ "$item" != "$vendor_name" ]; then
        new_items+=("$item")
    fi
done

chmod +w "$config_file"

if [ ${#new_items[@]} -eq 0 ]; then
    rm -f "$config_file"
    if [ $? -ne 0 ]; then
        log "[ERROR] Failed to delete $config_file."
        exit 1
    fi
else
    new_value=$(IFS=,; echo "${new_items[*]}")
    new_line="load_priority=$new_value"
    sed -i "s|^\s*load_priority\s*=.*|$new_line|" "$config_file"
    if [ $? -ne 0 ]; then
        log "[ERROR] Failed to update $config_file."
        exit 1
    fi
fi

log "[INFO] Successfully deleted $vendor_name."
exit 0