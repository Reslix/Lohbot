#ifndef __BYTEPASSING_H__
#define __BYTEPASSING_H__

enum commands {
				        STOP,
                FWD,
                BAK,
                LFT,
                RHT,
                DSTL,
                DSTM,
                DSTR
                };

#define BUF_LEN 80
typedef char message;
int pos = 0;
message messages[BUF_LEN];


int fill_buffer();
void run_commands();
void run_command(char);

#endif


