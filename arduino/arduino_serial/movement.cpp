/*
 * PWM motor controller library for arduino
 * Written for ENEE408i at UMD
 * Kyle Montemayor
 */

#include "Arduino.h"
#include "movement.hpp"

void Movement::turn(Movement::turn_dir dir, int speed){
    if(dir == Right){
        digitalWrite(left_forward, LOW);
        digitalWrite(left_backward, HIGH);

        digitalWrite(right_forward, HIGH);
        digitalWrite(right_backward, LOW);

        analogWrite(right_pwm, speed*SCALING);
        analogWrite(left_pwm, (speed*3)>>2); // Scaling factor between forward and back ~ 3/4,
                                             // So dividing by 4
    }else if (dir == Left) {
        digitalWrite(left_forward, HIGH);
        digitalWrite(left_backward, LOW);

        digitalWrite(right_forward, LOW);
        digitalWrite(right_backward, HIGH);

        analogWrite(right_pwm, (speed)*SCALING);
        analogWrite(left_pwm, (speed*3)>>2);
    }
}

void Movement::forward(int speed){
    if(speed < 0){
        digitalWrite(left_forward, LOW);
        digitalWrite(left_backward, HIGH);

        digitalWrite(right_forward, LOW);
        digitalWrite(right_backward, HIGH);

        analogWrite(right_pwm, -speed*SCALING);
        analogWrite(left_pwm, -speed);
    }else{
        digitalWrite(left_forward, HIGH);
        digitalWrite(left_backward, LOW);

        digitalWrite(right_forward, HIGH);
        digitalWrite(right_backward, LOW);
        analogWrite(right_pwm, speed*SCALING);
        analogWrite(left_pwm, speed);
    }
}

void Movement::stop(){
  Movement::forward(0);
}


