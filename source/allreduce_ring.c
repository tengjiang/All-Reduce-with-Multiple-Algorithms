/*
  Foundations of Parallel and Distributed Computing, Fall 2021.
  Instructor: Prof. Chao Yang @ Peking University.
  Implementer: Teng Jiang @ Peking University
  Date: 1/11/2021
*/

#include <mpi.h>
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

void reduce(float* dst, float* src, size_t size) {
  // Accumulate values from `src` into `dst` on the CPU.
  for(size_t i = 0; i < size; i++) {
    dst[i] += src[i];
    }
}

/*
void copy(float* dst, float* src, size_t size) {
  // CPU memory allocation through standard allocator.
  std::memcpy((void*) dst, (void*) src, size * sizeof(float));

}*/

  void RingAllreduce(float* data, size_t length) {
    
    const size_t size=comm_size;
    // Partition the elements of the array into N approximately equal-sized
    // chunks, where N is the MPI size.
    const size_t segment_size = length / size;
    size_t segment_sizes[size];
    for (size_t i = 0;i<size;++i){
      segment_sizes[i] = segment_size;
    }

    const size_t residual = length % size;
    for (size_t i = 0; i < residual; ++i) {
        segment_sizes[i]++;
    }

    // 尽量让每个segment的size一样，最多只差一个。
    // Compute where each chunk ends.
    size_t segment_ends[size];
    segment_ends[0] = segment_sizes[0];
    for (size_t i = 1; i < size; ++i) {
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    }

    // Allocate a temporary buffer to store incoming data.
    // We know that segment_sizes[0] is going to be the largest buffer size,
    // because if there are any overflow elements at least one will be added to
    // the first segment.

    float buffer[segment_sizes[0]];

    // Receive from your left neighbor with wrap-around.
    const size_t recv_from = (rank - 1 + size) % size;

    // Send to your right neighbor with wrap-around.
    const size_t send_to = (rank + 1) % size;

    MPI_Status recv_status;
    MPI_Request recv_req;
    MPI_Datatype datatype = MPI_FLOAT;

    // Now start ring. At every step, for every rank, we iterate through
    // segments with wraparound and send and recv from our neighbors and reduce
    // locally. At the i'th iteration, sends segment (rank - i) and receives
    // segment (rank - i - 1).

    for (int i = 0; i < size - 1; i++) {
        int recv_chunk = (rank - i - 1 + size) % size;
        int send_chunk = (rank - i + size) % size;
        float* segment_send = &(data[segment_ends[send_chunk] -
                                   segment_sizes[send_chunk]]);

        MPI_Irecv(buffer, segment_sizes[recv_chunk],
                datatype, recv_from, 0, MPI_COMM_WORLD, &recv_req);

        MPI_Send(segment_send, segment_sizes[send_chunk],
                MPI_FLOAT, send_to, 0, MPI_COMM_WORLD);

        float *segment_update = &(data[segment_ends[recv_chunk] -
                                         segment_sizes[recv_chunk]]);

        // Wait for recv to complete before reduction
        MPI_Wait(&recv_req, &recv_status);

        reduce(segment_update, buffer, segment_sizes[recv_chunk]);
    }

    // Now start pipelined ring allgather. At every step, for every rank, we
    // iterate through segments with wraparound and send and recv from our
    // neighbors. At the i'th iteration, rank r, sends segment (rank + 1 - i)
    // and receives segment (rank - i).
    
    for (size_t i = 0; i < size - 1; ++i) {
        int send_chunk = (rank - i + 1 + size) % size;
        int recv_chunk = (rank - i + size) % size;
        // Segment to send - at every iteration we send segment (r+1-i)
        float* segment_send = &(data[segment_ends[send_chunk] -
                                       segment_sizes[send_chunk]]);

        // Segment to recv - at every iteration we receive segment (r-i)
        float* segment_recv = &(data[segment_ends[recv_chunk] -
                                       segment_sizes[recv_chunk]]);
        MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
                datatype, send_to, 0, segment_recv,
                segment_sizes[recv_chunk], datatype, recv_from,
                0, MPI_COMM_WORLD, &recv_status);
    }
    
    }
  RingAllreduce(data,sizeof(data)/sizeof(data[0]));
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
  }
  if (rank == 0 && correct_count == comm_size) {
    printf("Buffer size: %d, comm size: %d\n", BUFFER_SIZE, comm_size);
    printf("Correct results.\n");
    printf("Your implementation wall time:%f\n", impl_time);
  }

  return 0;
}
