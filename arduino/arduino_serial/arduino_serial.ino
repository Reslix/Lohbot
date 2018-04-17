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
int SPEED = 60;
int sample = 10;


int debugArduinoSlave = 1;
#define DebugArduinoSlave(args...) if (debugArduinoSlave) Serial.print(args)
#define DebugArduinoSlaveln(args...) if (debugArduinoSlave) Serial.println(args)

void setup() {
  #ifdef DEBUG
  randomSeed(analogRead(0));
  #endif

  Serial.begin(115200);
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
                #ifndef DEBUG 
                Serial.write('s');
                #endif
                break;
            case 'f':
                while(!Serial.available()); // so we can read the next byte
                c = Serial.read();
                #ifndef DEBUG
                move.forward(c);
                #endif
                Serial.write(c);
                break;
            case 'b':
                while(!Serial.available()); // so we can read the next byte
                c = Serial.read(); //Error checking later
                #ifndef DEBUG
                move.forward(c);
                #endif
                Serial.write(c);
                break;
            case 'l':
                while(!Serial.available()); // so we can read the next byte
                c = Serial.read(); //Error checking later
                #ifndef DEBUG
                move.turn(Movement::turn_dir::Left, c);
                #endif
                Serial.write(c);
                break;
            case 'r':
                while(!Serial.available()); // so we can read the next byte
                c = Serial.read(); //Error checking later
                #ifndef DEBUG
                move.turn(Movement::turn_dir::Right, c);
                #endif
                Serial.write(c);
                break;
            case 'x':
                #ifdef DEBUG
                Serial.write(random(0, 150));
                #else
                Serial.write(ping.distance_left());
                #endif
                break;
            case 'y':
                #ifdef DEBUG
                Serial.write(random(0, 150));
                #else
                Serial.write(ping.distance_middle());
                #endif
                break;
            case 'z':
                #ifdef DEBUG
                Serial.write(random(0, 150));
                #elses
                Serial.write(ping.distance_right());
                #endif
                break;
            case 'd': // direct write
                while(!Serial.available());
                char dirs = Serial.read();
                while(!Serial.available());
                char speed_l = Serial.read();
                while(!Serial.available());
                char speed_r = Serial.read();
                
                move.direct(dirs, speed_l, speed_r);
                Serial.write(dirs^speed_l^speed_r); // 'hashing'
                break;
            default:
                Serial.write(c);
                move.stop();
                break;
        }
    }
}


