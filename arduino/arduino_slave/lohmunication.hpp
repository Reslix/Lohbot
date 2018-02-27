/*
 * Serial upstream/downstream communication
 * Kyle Montemayor
 */
#ifndef __LOHMUNICATION_H__
#define __LOHMUNICATION_H__

#include "movement.hpp"
/*
 * Want to have enums for the different message types, and then for data inside 
 * the messages. Because we use ints for data can jenk that instead of the enum
 * if needed
 */

enum commands {move, get_dist};

union Data {
    int i;
    float f;
} data;

typedef struct {
	commands command;
	data data;
} message;


/* Also want a queue for the commands, too lazy to implement separately */
// Kind of arbitrary but not the biggest deal

#define MESSAGE_QUEUE_LENGTH 128
static int queue_pos = 0;
static int new_commands = 0;

message message_queue[MESSAGE_QUEUE_LENGTH];

int fill_buffer();

void run_commands();

void run_command(message);
#endif

