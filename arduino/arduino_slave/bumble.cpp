
#include "movement.hpp"
#include "bumble.hpp"

Bumble::Bumble(Movement movement){

}
int MIN_DIST = 25;
int SPEED = 32;

void Bumble::run(){
    delay(sample);
    int * distances = ping3();
    int left_dist = distances[LEFT];
    int right_dist = distances[RIGHT];
    int mid_dist = distances[MIDDLE];

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

int Bumble::decide_direction(int left, int mid, int right){
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

int Bumble::decide_speed(int left, int mid, int right){
    if(mid <= MIN_DIST || left <= MIN_DIST || right <= MIN_DIST){
        return 0;
    }
    if()
}

int * Bumble::ping3(){

}