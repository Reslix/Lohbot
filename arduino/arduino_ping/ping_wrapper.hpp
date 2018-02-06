/*
 * Wrapper over NewPing library
 * Written for ENEE408i at UMD
 */
#ifndef __PING_WRAPPER_H__
#define __PING_WRAPPER_H__

#include <NewPing.h>

class PingPing{
    private:
        int left_pin;
        int middle_pin;
        int right_pin;
        /// Obstacles above this distance are reported as clear
        int max_dist;
        NewPing left_ping;
        NewPing middle_ping;
        NewPing right_ping;

    public:
        int distance_left();
        int distance_middle();
        int distance_right();

        PingPing() {
            left_pin = 4;
            middle_pin = 6;
            right_pin = 7;
            max_dist = 200;
            left_ping = NewPing(left_pin, left_pin, 200);
            middle_ping = NewPing(middle_pin, middle_pin, 200);
            right_ping = NewPing(right_pin, right_pin, 200);
        }
};

#endif
