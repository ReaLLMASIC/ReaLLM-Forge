template<typename T>
using InputChannel = __xls_channel<T, __xls_channel_dir_In>;
template<typename T>
using OutputChannel = __xls_channel<T, __xls_channel_dir_Out>;

class TestBlock {
public:
    InputChannel<int> size;
    InputChannel<int> A;
    InputChannel<int> B;
    OutputChannel<int> C;
    
    #pragma hls_top
    void Run() {
        const int MAX_SIZE = 10;
        int m = MAX_SIZE;
        int n = MAX_SIZE;
        int p = MAX_SIZE;
        int A_local[MAX_SIZE][MAX_SIZE];
        int B_local[MAX_SIZE][MAX_SIZE];
        int C_local[MAX_SIZE][MAX_SIZE];                  // (m, n) * (n, p) = (m, p)

        #pragma hls_unroll
        for (unsigned int i = 0; i < m; ++i) {
            #pragma hls_unroll
            for (unsigned int j = 0; j < n; ++j)
                A_local[i][j] = A.read();
        }
        #pragma hls_unroll
        for (unsigned int i = 0; i < n; ++i) {
            #pragma hls_unroll
            for (unsigned int j = 0; j < p; ++j)
                B_local[i][j] = B.read();
        }
        
        #pragma hls_unroll
        for (unsigned int i = 0; i < m; ++i) {
            #pragma hls_unroll
            for (unsigned int j = 0; j < p; ++j) {
                C_local[i][j] = 0;
                #pragma hls_unroll
                for (unsigned int k = 0; k < n; ++k)
                    C_local[i][j] += A_local[i][k] * B_local[k][j];
            }
        }
    
        #pragma hls_unroll
        for (unsigned int i = 0; i < m; ++i) {
            #pragma hls_unroll
            for (unsigned int j = 0; j < p; ++j)
                C.write(C_local[i][j]);
        }
    }
};
