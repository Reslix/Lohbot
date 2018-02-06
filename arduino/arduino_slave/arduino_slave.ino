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
  delay(2000);
  Serial.println("going backwards");
  move.forward(-128);
  delay(200);
  move.stop();
  delay(3000);

  Serial.println("going forwards");
  move.forward(128);
  delay(200);
  move.stop();
  delay(3000);


  Serial.println("turning left");
  move.turn(0);
  delay(200);
  move.stop();
  delay(3000);


  Serial.println("turning right");
  move.turn(1);
  delay(200);
  move.stop();
  delay(3000);

  

}
