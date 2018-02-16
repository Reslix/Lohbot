/*
 * Testing ping code
 */

#include "ping_wrapper.hpp"

PingPing pingping;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

}

void loop() {
  // put your main code here, to run repeatedly:
  delay(2000);

  int obstacle_val;
  obstacle_val = pingping.distance_left();
  Serial.print("left: ");
  Serial.println(obstacle_val);

 obstacle_val = pingping.distance_middle();
 Serial.print("center: ");
 Serial.println(obstacle_val);

 obstacle_val = pingping.distance_right();
 Serial.print("right: ");
 Serial.println(obstacle_val);
}
