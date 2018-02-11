/*
 * Arduino code for 408i
 * 
 */

#include "movement.hpp"
#include "ping_wrapper.hpp"

Movement move;
PingPing ping;

int MIN_DIST = 25;
int SPEED = 32;
int sample = 100;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

}

int decide_direction(int left, int mid, int right){
    if(mid <= MIN_DIST || left <= MIN_DIST || right <= MIN_DIST){
        if(mid > left && mid > right && mid > MIN_DIST){
            return -1;
        }
        if(left < right){
            return 1;
        }
        if(right < left){
            return 0;
        }
    }
    return -1;
}

void loop() {
    // put your main code here, to run repeatedly:
    delay(sample);
    int * distances = ping3();
    int left_dist = distances[0];
    int right_dist = distances[2];
    int mid_dist = distances[1];

    int dir = decide_direction(left_dist, mid_dist, right_dist);

    if(dir != -1){
        move.forward(0);
        if(dir == 0){
            move.turn(1);
        }else{
            move.turn(0);
        }
    }else{
        move.forward(SPEED);
    }
}

int * ping3(){
    int distances[3];

    distances[0] = ping.distance_left();
    distances[1] = ping.distance_middle();
    distances[2] = ping.distance_right();

    return distances;
}
