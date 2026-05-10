#!/bin/bash

DEFAULT_CAN_NAME="${1:-can0}"
DEFAULT_BITRATE="${2:-1000000}"
USB_ADDRESS="${3}"

echo "-------------------START-----------------------"

if ! dpkg -l | grep -q "ethtool"; then
    echo "\e[31mError: ethtool not detected in the system.\e[0m"
    echo "Please use the following command to install ethtool:"
    echo "sudo apt update && sudo apt install ethtool"
    exit 1
fi

if ! dpkg -l | grep -q "can-utils"; then
    echo "\e[31mError: can-utils not detected in the system.\e[0m"
    echo "Please use the following command to install can-utils:"
    echo "sudo apt update && sudo apt install can-utils"
    exit 1
fi

echo "Both ethtool and can-utils are installed."

get_bus_info() {
    local iface="$1"
    sudo ethtool -i "$iface" 2>/dev/null | awk '/bus-info/ {print $2}'
}

get_operstate() {
    local iface="$1"
    local state_file="/sys/class/net/${iface}/operstate"
    if [ -r "$state_file" ]; then
        cat "$state_file"
    else
        echo "unknown"
    fi
}

ensure_interface_up() {
    local iface="$1"
    local target_bitrate="$2"
    local attempts=0
    local operstate

    while [ "$attempts" -lt 2 ]; do
        operstate="$(get_operstate "$iface")"
        if [ "$operstate" = "up" ]; then
            return 0
        fi

        attempts=$((attempts + 1))
        echo "Interface $iface operstate is '$operstate'. Forcing CAN reset (attempt ${attempts}/2)."
        sudo ip link set "$iface" down || true
        sudo ip link set "$iface" type can bitrate "$target_bitrate"
        sudo ip link set "$iface" up
        sleep 1
    done

    operstate="$(get_operstate "$iface")"
    if [ "$operstate" != "up" ]; then
        echo "\e[31mError: interface $iface failed to reach operstate 'up' (current: $operstate).\e[0m"
        return 1
    fi
}

configure_interface() {
    local iface="$1"
    local target_name="$2"
    local bus_info
    local is_link_up
    local current_bitrate

    bus_info=$(get_bus_info "$iface")
    if [ -z "$bus_info" ]; then
        bus_info="unknown"
    fi

    echo "Configuring interface $iface (USB port: $bus_info)."

    is_link_up=$(ip link show "$iface" | grep -q "UP" && echo "yes" || echo "no")
    current_bitrate=$(ip -details link show "$iface" | grep -oP 'bitrate \K\d+' | head -n 1)
    current_bitrate="${current_bitrate:-0}"

    if [ "$is_link_up" = "yes" ] && [ "$current_bitrate" = "$DEFAULT_BITRATE" ]; then
        echo "Interface $iface is already activated with bitrate $DEFAULT_BITRATE."
    else
        if [ "$is_link_up" = "yes" ]; then
            echo "Interface $iface is already up, but bitrate is $current_bitrate. Resetting it to $DEFAULT_BITRATE."
        else
            echo "Interface $iface is down or bitrate is not set. Activating it with bitrate $DEFAULT_BITRATE."
        fi

        sudo ip link set "$iface" down
        sudo ip link set "$iface" type can bitrate "$DEFAULT_BITRATE"
        sudo ip link set "$iface" up
        echo "Interface $iface has been activated."
    fi

    if ! ensure_interface_up "$iface" "$DEFAULT_BITRATE"; then
        echo "\e[31mError: CAN interface $iface on USB port $bus_info is present but not usable.\e[0m"
        exit 1
    fi

    if [ "$target_name" != "$iface" ]; then
        if ip link show "$target_name" >/dev/null 2>&1; then
            echo "\e[31mError: target interface name $target_name already exists.\e[0m"
            exit 1
        fi

        echo "Renaming interface $iface to $target_name."
        sudo ip link set "$iface" down
        sudo ip link set "$iface" name "$target_name"
        sudo ip link set "$target_name" up
        if ! ensure_interface_up "$target_name" "$DEFAULT_BITRATE"; then
            echo "\e[31mError: renamed CAN interface $target_name on USB port $bus_info is not usable.\e[0m"
            exit 1
        fi
        echo "The interface has been renamed to $target_name and reactivated."
    fi
}

mapfile -t INTERFACES < <(ip -br link show type can | awk '{print $1}')
CURRENT_CAN_COUNT="${#INTERFACES[@]}"

if [ "$CURRENT_CAN_COUNT" -eq 0 ]; then
    echo "\e[31mError: Unable to detect any CAN interface.\e[0m"
    echo "-------------------ERROR-----------------------"
    exit 1
fi

if [ -n "$USB_ADDRESS" ]; then
    echo "Detected USB hardware address parameter: $USB_ADDRESS"

    INTERFACE_NAME=""
    for iface in "${INTERFACES[@]}"; do
        BUS_INFO=$(get_bus_info "$iface")
        if [ "$BUS_INFO" = "$USB_ADDRESS" ]; then
            INTERFACE_NAME="$iface"
            break
        fi
    done

    if [ -z "$INTERFACE_NAME" ]; then
        echo "\e[31mError: Unable to find CAN interface corresponding to USB hardware address $USB_ADDRESS.\e[0m"
        echo "-------------------ERROR-----------------------"
        exit 1
    fi

    echo "Found interface $INTERFACE_NAME for USB port $USB_ADDRESS."
    configure_interface "$INTERFACE_NAME" "$DEFAULT_CAN_NAME"
else
    if [ "$CURRENT_CAN_COUNT" -eq 1 ]; then
        echo "Detected 1 CAN interface. Activating it."
    else
        echo "Detected $CURRENT_CAN_COUNT CAN interfaces. Activating all of them."
        echo "Tip: if you want to configure only one interface, run:"
        echo "bash can_activate.sh can0 1000000 <usb-bus-info>"
    fi

    for iface in "${INTERFACES[@]}"; do
        configure_interface "$iface" "$iface"
    done
fi

echo "-------------------OVER------------------------"
