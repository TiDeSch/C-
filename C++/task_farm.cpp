/*
  Assignment: Make an MPI task farm. A "task" is a randomly generated integer.
  To "execute" a task, the worker sleeps for the given number of milliseconds.
  The result of a task should be send back from the worker to the master. It
  contains the rank of the worker
*/

#include <iostream>
#include <random>
#include <chrono>
#include <thread>
#include <array>

// To run an MPI program we always need to include the MPI headers
#include <mpi.h>

const int NTASKS = 5000;  // number of tasks
const int RANDOM_SEED = 1234;

// call this function to complete the task. It sleeps for task milliseconds
void task_function(int task) {
    std::this_thread::sleep_for(std::chrono::milliseconds(task));
}

void master(int nworker) {
    std::array<int, NTASKS> task, result;

    // set up a random number generator
    std::random_device rd;
    std::default_random_engine engine;
    engine.seed(RANDOM_SEED);
    // make a distribution of random integers in the interval [0:30]
    std::uniform_int_distribution<int> distribution(0, 30);

    for (int& t : task) {
        t = distribution(engine);   // set up some "tasks"
    }

    if (nworker == 0) {
        // No workers available, handle all tasks in the master
        for (int i = 0; i < NTASKS; i++) {
            task_function(task[i]);
            result[i] = 0;  // Master's rank is 0
        }
    } else {
        // Non-blocking communication buffers
        MPI_Request send_requests[NTASKS];
        MPI_Request recv_requests[NTASKS];
        int task_index = 0;

        // Send initial tasks to all workers
        for (int worker = 1; worker <= nworker; worker++) {
            MPI_Isend(&task[task_index], 1, MPI_INT, worker, 0, MPI_COMM_WORLD, &send_requests[task_index]);
            MPI_Irecv(&result[task_index], 1, MPI_INT, worker, 0, MPI_COMM_WORLD, &recv_requests[task_index]);
            task_index++;
        }

        // Receive results and distribute remaining tasks
        while (task_index < NTASKS) {
            // Wait for any result to arrive
            int completed_index;
            MPI_Status status;
            MPI_Waitany(task_index, recv_requests, &completed_index, &status);

            // Send a new task to the worker that just completed a task
            MPI_Isend(&task[task_index], 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &send_requests[task_index]);
            MPI_Irecv(&result[task_index], 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &recv_requests[task_index]);
            task_index++;
        }

        // Wait for all remaining results
        MPI_Waitall(NTASKS, recv_requests, MPI_STATUSES_IGNORE);

        // Send termination signal to all workers
        for (int worker = 1; worker <= nworker; worker++) {
            int termination_signal = -1;
            MPI_Send(&termination_signal, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
        }
    }

    // Print out a status on how many tasks were completed by each worker
    for (int worker = 0; worker <= nworker; worker++) {
        int tasksdone = 0;
        int workdone = 0;
        for (int itask = 0; itask < NTASKS; itask++) {
            if (result[itask] == worker) {
                tasksdone++;
                workdone += task[itask];
            }
        }
        if (worker == 0) {
            std::cout << "Master: Master solved " << tasksdone << " tasks\n";
        } else {
            std::cout << "Master: Worker " << worker << " solved " << tasksdone << " tasks\n";
        }
    }
}

void worker(int rank) {
    int task;
    MPI_Status status;

    while (true) {
        // Receive a task from the master
        MPI_Recv(&task, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        // Check for termination signal
        if (task == -1) {
            break;
        }

        // Execute the task
        task_function(task);

        // Send the result back to the master (non-blocking)
        MPI_Request send_request;
        MPI_Isend(&rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &send_request);
        MPI_Wait(&send_request, MPI_STATUS_IGNORE); // Wait for the send to complete
    }
}

int main(int argc, char *argv[]) {
    int nrank, rank;

    MPI_Init(&argc, &argv);                // set up MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nrank); // get the total number of ranks
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // get the rank of this process

    if (rank == 0)       // rank 0 is the master
        master(nrank - 1); // there is nrank-1 worker processes
    else                 // ranks in [1:nrank] are workers
        worker(rank);

    MPI_Finalize();      // shutdown MPI
}