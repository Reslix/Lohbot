



class Bumble(){

    private:
        int MIN_DIST = 40;

        enum sensor = {LEFT, MIDDLE, RIGHT}

        int * ping3();
        int decide_direction(int, int, int);

    public:
        void run();
}