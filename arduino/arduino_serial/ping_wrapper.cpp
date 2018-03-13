/*
 * Wrapper over NewPing library
 */

#include "Arduino.h"
#include "ping_wrapper.hpp"

/// Returns the distance to the nearest obstruction on the left in cm
int PingPing::distance_left() {
    delay(50);
    int dist = left_ping.ping_cm();
    return dist > 0 ? dist : 201;
}

int PingPing::distance_middle() {
    delay(50);
    int dist = middle_ping.ping_cm();
    return dist > 0 ? dist : 201;
}

int PingPing::distance_right() {
    delay(50);
    int dist = right_ping.ping_cm();
    return dist > 0 ? dist : 201;
}

