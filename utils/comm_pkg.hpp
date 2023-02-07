#ifndef SOLID_COMM_PKG_HPP
#define SOLID_COMM_PKG_HPP

#include "common.hpp"

class StructCommPackage {
public:
    MPI_Comm cart_comm = MPI_COMM_NULL;
    int cart_ids[3];
    int my_pid;
    bool relay = false;// 是否是接力（三踢脚）的方式进行通信
    int ngbs_pid[NUM_NEIGHBORS];
    MPI_Datatype send_subarray[NUM_NEIGHBORS], recv_subarray[NUM_NEIGHBORS];
    mutable MPI_Request send_req[NUM_NEIGHBORS], recv_req[NUM_NEIGHBORS];
    MPI_Datatype mpi_scalar_type;

    StructCommPackage(bool relay_mode): relay(relay_mode) {}
    void exec_comm(void * data);
    void wait();
    ~StructCommPackage();
};

void StructCommPackage::exec_comm(void * data) {
    // printf("ngb id %d %d %d %d %d %d\n", ngbs_pid[K_L], ngbs_pid[K_U], ngbs_pid[I_L], ngbs_pid[I_U], ngbs_pid[J_L], ngbs_pid[J_U]);
    if (relay) {
        MPI_Status status;
        // 先收发k向的
        MPI_Sendrecv(   data, 1, send_subarray[K_L], ngbs_pid[K_L], 111,
                        data, 1, recv_subarray[K_U], ngbs_pid[K_U], 111, cart_comm, &status);
        MPI_Sendrecv(   data, 1, send_subarray[K_U], ngbs_pid[K_U], 112,
                        data, 1, recv_subarray[K_L], ngbs_pid[K_L], 112, cart_comm, &status);
        // 二踢脚：收发i向的
        MPI_Sendrecv(   data, 1, send_subarray[I_L], ngbs_pid[I_L], 121,
                        data, 1, recv_subarray[I_U], ngbs_pid[I_U], 121, cart_comm, &status);
        MPI_Sendrecv(   data, 1, send_subarray[I_U], ngbs_pid[I_U], 122,
                        data, 1, recv_subarray[I_L], ngbs_pid[I_L], 122, cart_comm, &status);
        // 三踢脚：收发j向的
        MPI_Sendrecv(   data, 1, send_subarray[J_L], ngbs_pid[J_L], 131,
                        data, 1, recv_subarray[J_U], ngbs_pid[J_U], 131, cart_comm, &status);
        MPI_Sendrecv(   data, 1, send_subarray[J_U], ngbs_pid[J_U], 132,
                        data, 1, recv_subarray[J_L], ngbs_pid[J_L], 132, cart_comm, &status);
    }
    else {
        for (int ip = 0; ip < NUM_NEIGHBORS; ip++) {
            // int tag = my_pid + ngbs_pid[ip];
            MPI_Isend(data, 1, send_subarray[ip], ngbs_pid[ip], 11, cart_comm, &send_req[ip]);
            MPI_Irecv(data, 1, recv_subarray[ip], ngbs_pid[ip], 11, cart_comm, &recv_req[ip]);
        }
        wait();
    }
}

void StructCommPackage::wait() {
    assert(relay == false);
    MPI_Waitall(NUM_NEIGHBORS, send_req, MPI_STATUS_IGNORE);
    MPI_Waitall(NUM_NEIGHBORS, recv_req, MPI_STATUS_IGNORE);
}

StructCommPackage::~StructCommPackage()
{
    for (int ingb = 0; ingb < NUM_NEIGHBORS; ingb++) {
        if (send_subarray[ingb] != MPI_DATATYPE_NULL)
            MPI_Type_free(&send_subarray[ingb]);
        if (recv_subarray[ingb] != MPI_DATATYPE_NULL)
            MPI_Type_free(&recv_subarray[ingb]);
    }
}


#endif