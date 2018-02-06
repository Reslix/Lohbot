/*
 * PWM motor controller library for arduino
 * Written for ENEE408i at UMD
 * Kyle Montemayor
 */

#include "Arduino.h"
#include "movement.hpp"

void Movement::turn(int dir){
    if(dir == Left){
        digitalWrite(left_forward, LOW);
        digitalWrite(left_backward, HIGH);

        digitalWrite(right_forward, HIGH);
        digitalWrite(right_backward, LOW);

        analogWrite(right_pwm, 32);
        analogWrite(left_pwm, 48);
    }else{
        digitalWrite(left_forward, HIGH);
        digitalWrite(left_backward, LOW);

        digitalWrite(right_forward, LOW);
        digitalWrite(right_backward, HIGH);

        analogWrite(right_pwm, 48);
        analogWrite(left_pwm, 32);
    }   
}

void Movement::forward(int speed){
    if(speed < 0){
        digitalWrite(left_forward, LOW);
        digitalWrite(left_backward, HIGH);

        digitalWrite(right_forward, LOW);
        digitalWrite(right_backward, HIGH);

        analogWrite(right_pwm, speed);
        analogWrite(left_pwm, speed);
    }else{
        digitalWrite(left_forward, HIGH);
        digitalWrite(left_backward, LOW);

        digitalWrite(right_forward, HIGH);
        digitalWrite(right_backward, LOW);

        analogWrite(right_pwm, speed);
        analogWrite(left_pwm, speed);
    }
}

void Movement::stop(){
  Movement::forward(0);
}

