/*
 * PWM motor controller library for arduino
 * Written for ENEE408i at UMD
 * Kyle Montemayor
 */
#include "Arduino.h"
#ifndef __MOVEMENT_H__
#define __MOVEMENT_H__

#define SCALING 0.85
class Movement{
    private:
        int left_forward;
        int left_backward;
        int left_pwm;
        int left_dir;

        int right_forward;
        int right_backward;
        int right_pwm;
        int right_dir;

    public:
        enum turn_dir {Left, Right, Forward, Backward};
        void turn(Movement::turn_dir);
        void forward(int);
        void stop(void);

        Movement(){
            left_forward = 8;
            left_backward = 9;
            left_pwm = 3;
            left_dir = 1;

            right_forward = 10;
            right_backward = 11;
            right_pwm = 5;
            right_dir = 1;

            pinMode(left_forward, OUTPUT);
            pinMode(left_backward, OUTPUT);
            pinMode(left_pwm, OUTPUT);


            pinMode(right_forward, OUTPUT);
            pinMode(right_backward, OUTPUT);
            pinMode(right_pwm, OUTPUT);

        }

        Movement(int lf, int lb, int lp,
                 int rf, int rb, int rp){
            left_forward = lf;
            left_backward = lb;
            left_pwm = lp;

            right_forward = rf;
            right_backward = rb;
            right_pwm = rp;
        }
};

#endif


