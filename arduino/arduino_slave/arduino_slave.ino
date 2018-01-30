/*
 * Arduino code for 408i
 * 
 */


//Pins
const int left_forward = 10;
const int left_backward = 11;
const int left_pwm = 5;

const int right_forward = 8;
const int right_backward = 9;
const int right_pwm = 3;


void setup() {
  // put your setup code here, to run once:
  pinMode(left_forward, OUTPUT);
  pinMode(left_backward, OUTPUT);
  pinMode(right_forward, OUTPUT);
  pinMode(right_backward, OUTPUT);

}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(left_forward, LOW);
  digitalWrite(left_backward, HIGH);

  digitalWrite(right_forward, LOW);
  digitalWrite(right_backward, HIGH);


  for(int i = 0; i < 255; i++){
    analogWrite(left_pwm, i);
    analogWrite(right_pwm, i);

    delay(250);
  }
  

}
