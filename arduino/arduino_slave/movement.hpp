/*
 * PWM motor controller library for arduino
 * Written for ENEE408i at UMD
 * Kyle Montemayor
 */
#ifndef __MOVEMENT_H__
#define __MOVEMENT_H__

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
        void turn(int);
        void forward(int);

        enum turn_dir = {Left, Right};
        
        Movement(){
            left_forward = 8;
            left_backward = 9;
            left_pwm = 3;
            left_dir = 1;

            right_forward = 10;
            right_backward = 11;
            right_pwm = 5;
            right_dir = 1;
        }

        Movement(int lf, int lb, int lp, int ld,
                 int rf, int rb, int rp, int rd){
            left_forward = lf;
            left_backward = lb;
            left_pwm = lp;
            left_dir = ld;

            right_forward = rf;
            right_backward = rb;
            right_pwm = rp;
            right_dir = rd;
        }

#endif
