/*
 * Wrapper over NewPing library
 */

#include "Arduino.h"
#include "ping_wrapper.hpp"

/// Returns the distance to the nearest obstruction on the left in cm
int PingPing::distance_left(int threshold) {
    delay(50);
    return left_ping.ping_cm();
}

int PingPing::distance_middle(int threshold) {
    delay(50);
    return middle_ping.ping_cm();
}

int PingPing::distance_right(int threshold) {
    delay(50);
    return right_ping.ping_cm();
}