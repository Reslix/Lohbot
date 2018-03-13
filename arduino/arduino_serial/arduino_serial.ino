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
  Serial.flush();
  Serial.write('S');
  
}

void loop() {
    char c;
    if (Serial.available() > 0){
        c = Serial.read();
        // I know switches aren't the best...
        switch(c){
            case 's':
                move.stop();
                Serial.write('s');
                break;
            case 'f':
                move.forward(SPEED);
                Serial.write('f');
                break;
            case 'b':
                move.forward(-SPEED);
                Serial.write('b');
                break;
            case 'l':
                move.turn(Movement::turn_dir::Left);
                Serial.write('l');
                break;
            case 'r':
                move.turn(Movement::turn_dir::Right);
                Serial.write('r');
                break;
            case 'x':
                Serial.write(ping.distance_left());
                break;
            case 'y':
                Serial.write(ping.distance_middle());
                break;
            case 'z':
                Serial.write(ping.distance_right());
                break;
            default:
                Serial.write('E');
                break;
        }
    }
}

