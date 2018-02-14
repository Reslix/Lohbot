/*
 * Arduino code for 408i
 * Moves, avoiding obstacles
 */

#include "movement.hpp"
#include "ping_wrapper.hpp"

Movement move;
PingPing ping;

int MIN_DIST = 12;
int SPEED = 0;
int sample = 1000;

int debugArduinoSlave = 1;
#define DebugArduinoSlave(args...) if (debugArduinoSlave) Serial.print(args)
#define DebugArduinoSlaveln(args...) if (debugArduinoSlave) Serial.println(args)

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

}

Movement::turn_dir decide_direction(int left, int mid, int right){
    if(mid <= MIN_DIST || left <= MIN_DIST || right <= MIN_DIST){
        if(mid > left && mid > right && mid > MIN_DIST){
            return Movement::Forward;
        }
        if(left < right){
            return Movement::Right;
        }
        if(right < left){
            return Movement::Left;
        }
    }
    return Movement::Forward;
}

void loop() {
    // put your main code here, to run repeatedly:
    delay(sample);
    int left_dist = ping.distance_left();
    int right_dist = ping.distance_right();
    int mid_dist = ping.distance_middle();
    DebugArduinoSlave("left: ");
    DebugArduinoSlave(left_dist);
    DebugArduinoSlave("; right: ");
    DebugArduinoSlave(right_dist);
    DebugArduinoSlave("; middle: ");
    DebugArduinoSlaveln(mid_dist);

    Movement::turn_dir dir = decide_direction(left_dist, mid_dist, right_dist);

    if(dir != Movement::Forward){
        DebugArduinoSlave("Turning in direction: ");
        DebugArduinoSlaveln(dir);
        move.stop();
        delay(100);
        move.turn(dir);
    }else{
        DebugArduinoSlave("Going forward.");
        DebugArduinoSlaveln();
        move.stop();
        delay(100);
        move.forward(SPEED);
    }
}
