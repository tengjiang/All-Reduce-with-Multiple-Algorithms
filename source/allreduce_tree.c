/*
  Foundations of Parallel and Distributed Computing, Fall 2021.
  Instructor: Prof. Chao Yang @ Peking University.
  Implementer: Teng Jiang @ Peking University
  Date: 1/11/2021
*/

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BUFFER_SIZE (4 * 1024 * 1024)
#define ABS(x) (((x) > 0) ? (x) : -(x))
#define EPS 1e-5

double get_walltime() {
#if 1
  return MPI_Wtime();
#else
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
#endif
}

void initialize(int rank, float* data, int n) {
  int i = 0;
  srand(rank);
  for (i = 0; i < n; ++i) {
    data[i] = rand() / (float)RAND_MAX;
  }
}

int result_check(float* a, float* b, int n) {
  int i = 0;
  for (i = 0; i < n; ++i) {
    if (ABS(a[i] - b[i]) > EPS) {
      return 0;
    }
  }
  return 1;
}

void reduce(float* dst, float* src, size_t size) {
  // Accumulate values from `src` into `dst` on the CPU.
  for(size_t i = 0; i < size; i++) {
    dst[i] += src[i];
    }
}


int main(int argc, char* argv[]) {
  int rank, comm_size;

  float data[BUFFER_SIZE];
  float base_output[BUFFER_SIZE];
  double time0, time1;
  double impl_time = 0;
  int correct_count = 0, correct = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // initialization
  initialize(rank, data, BUFFER_SIZE);

  // ground true results
  MPI_Allreduce(data, base_output, BUFFER_SIZE, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  time0 = get_walltime();
  /* write your codes here */
  void TreeAllreduce(float* data, size_t length){
    int size=comm_size;

    // Partition the elements of the array into N approximately equal-sized
    // chunks, where N is the MPI size.

    const size_t segment_size = length / size;
    size_t segment_sizes[size];
    for (size_t i = 0;i<size;++i){
      segment_sizes[i] = segment_size;
    } // 记录每块的大小

    const size_t residual = length % size;
    for (size_t i = 0; i < residual; ++i) {
        segment_sizes[i]++;
    }

    // 尽量让每个segment的size一样，最多只差一个。
    // 计算每块结束的地方

    size_t segment_ends[size];
    segment_ends[0] = segment_sizes[0];
    for (size_t i = 1; i < size; ++i) {
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    }

    float buffer[segment_sizes[0]];

    MPI_Status recv_status;
    MPI_Request recv_req;
    MPI_Datatype datatype = MPI_FLOAT;
    int layer = (int)(log2(size));
    int reduced_num = (int)(pow(2,layer));


    // reduce
    
    if(reduced_num!=size){//size数不是2的幂，则先化成2的幂次
      if(rank>=reduced_num){ //先把后面余下的加到前面的，凑成2的幂次
        for(size_t i = 0;i<size;i++){
          float* segment_send = &(data[segment_ends[i] - segment_sizes[i]]);
          MPI_Send(segment_send, segment_sizes[i],
                MPI_FLOAT, rank-reduced_num,i+size, MPI_COMM_WORLD);
  
        }
      }
      else if(rank<size-reduced_num){ //先把后面余下的加到前面的，凑成2的幂次
        for(size_t i = 0;i<size;i++){
          float* segment_recv = &(data[segment_ends[i] - segment_sizes[i]]);
          MPI_Recv(buffer, segment_sizes[i],
                MPI_FLOAT, rank+reduced_num,i+size, MPI_COMM_WORLD,&recv_status);
          reduce(segment_recv, buffer, segment_sizes[i]);
     
        }
      }
      else{

      }
    }
    if(rank<reduced_num){

      
      int remain = reduced_num;
      int half = reduced_num;
      while(remain != 1){
        half = remain/2;
        if(rank < half){
          for(size_t i=0; i<size; i++){
            float* segment_recv = &(data[segment_ends[i] - segment_sizes[i]]);
            MPI_Recv(buffer, segment_sizes[i],MPI_FLOAT,rank+half,i, MPI_COMM_WORLD,&recv_status);
            reduce(segment_recv, buffer, segment_sizes[i]);
          }
        }
        else if(rank >= half){
          for(size_t i=0; i<size; i++){
            float* segment_send = &(data[segment_ends[i] - segment_sizes[i]]);
            MPI_Send(segment_send, segment_sizes[i],
                  MPI_FLOAT, rank-half,i, MPI_COMM_WORLD);
          }
          remain = half;
          break;
        }
        remain = half;
      }

    //broadcast完全相反

    while(remain != reduced_num){
      half = remain;
      remain = remain * 2;
      if(rank < half){
        for(size_t i=0; i<size; i++){
          float* segment_send = &(data[segment_ends[i] - segment_sizes[i]]);
          MPI_Send(segment_send, segment_sizes[i],
                MPI_FLOAT, rank+half,i, MPI_COMM_WORLD);
        }
      }else if(rank >= half && rank < remain){
        for(size_t i=0; i<size; i++){
          float* segment_recv = &(data[segment_ends[i] - segment_sizes[i]]);
          MPI_Recv(segment_recv, segment_sizes[i],MPI_FLOAT,rank-half,i, MPI_COMM_WORLD,&recv_status);
        }
      }
    }
    }
  
    if(reduced_num!=size){//size数不是2的幂，则先化成2的幂次
      if(rank>=reduced_num){ 
        for(size_t i = 0;i<size;i++){
          float* segment_recv = &(data[segment_ends[i] - segment_sizes[i]]);
          MPI_Recv(segment_recv, segment_sizes[i],
                MPI_FLOAT, rank-reduced_num,i+2*size, MPI_COMM_WORLD,&recv_status);
     
        }
      }
      else if(rank<size-reduced_num){
        for(size_t i = 0;i<size;i++){
          float* segment_send = &(data[segment_ends[i] - segment_sizes[i]]);
          MPI_Send(segment_send, segment_sizes[i],
                MPI_FLOAT, rank+reduced_num,i+2*size, MPI_COMM_WORLD);
         
        }
      }
     
    }
  }

  TreeAllreduce(data,sizeof(data)/sizeof(data[0]));
  time1 = get_walltime() - time0;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&time1, &impl_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  // check correctness and report results
  correct = result_check(base_output, data, BUFFER_SIZE);
  MPI_Reduce(&correct, &correct_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Finalize();

  if (!correct) {
    printf("Wrong answer on rank %d.\n", rank);
    printf("%f%f     ",data[0],data[1]);
    printf("%f%f     ",base_output[0],base_output[1]);
  }
  if (rank == 0 && correct_count == comm_size) {
    printf("Buffer size: %d, comm size: %d\n", BUFFER_SIZE, comm_size);
    printf("Correct results.\n");
    printf("Your implementation wall time:%f\n", impl_time);
  }

  return 0;
}
