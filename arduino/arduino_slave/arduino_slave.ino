/*
 * Arduino code for 408i
 * 
 */

#include "movement.hpp"

Movement move;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

}

void loop() {
  // put your main code here, to run repeatedly:
  move.turn(1);
  

}
