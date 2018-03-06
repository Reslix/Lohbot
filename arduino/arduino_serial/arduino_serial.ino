/*
 * Arduino code for 408i
 * Moves, avoiding obstacles
 */

#include "movement.hpp"
#include "ping_wrapper.hpp"
#include "byte_passing.h"

Movement move;
PingPing ping;

int MIN_DIST = 30;
int SPEED = 48;
int sample = 10;

int debugArduinoSlave = 1;
#define DebugArduinoSlave(args...) if (debugArduinoSlave) Serial.print(args)
#define DebugArduinoSlaveln(args...) if (debugArduinoSlave) Serial.println(args)

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

}

void loop() {
    char c;
    while (Serial.available() > 0){
        c = Serial.read();
        switch(c){
            case STOP:
                move.stop();
                Serial.write(1);
                break;
            case FWD:
                move.forward(SPEED);
                Serial.write(1);
                break;
            case BAK:
                move.forward(-SPEED);
                Serial.write(1);
                break;
            case LFT:
                move.turn(Movement::turn_dir::Left);
                Serial.write(1);
                break;
            case RHT:
                move.turn(Movement::turn_dir::Right);
                Serial.write(1);
                break;
            case DSTL:
                Serial.write(ping.distance_left());
                break;
            case DSTM:
                Serial.write(ping.distance_middle());
                break;
            case DSTR:
                Serial.write(ping.distance_right());
                break;
            default:
                Serial.write(-1);
                break;
        }

    }
}
