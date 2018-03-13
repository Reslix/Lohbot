#include "NewPing.h"

/*
 * Wrapper over NewPing library
 * Written for ENEE408i at UMD
 */
#ifndef __PING_WRAPPER_H__
#define __PING_WRAPPER_H__

class PingPing{
    private:
        uint8_t left_pin = 2;
        uint8_t middle_pin = 6;
        uint8_t right_pin = 7;
        // Obstacles above this distance are reported as clear
        int max_dist;
        NewPing left_ping;
        NewPing middle_ping;
        NewPing right_ping;

    public:
        int distance_left();
        int distance_middle();
        int distance_right();

        PingPing() : left_ping(left_pin, left_pin, 200), middle_ping(middle_pin, middle_pin, 200),
              right_ping(right_pin, right_pin, 200) {
            max_dist = 200;
        }
};

#endif

