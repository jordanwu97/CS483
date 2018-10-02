#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 32

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // Identify the row and column of the P element to work on
  int Row = by * blockDim.x + ty;
  int Col = bx * blockDim.y + tx;
  float Pvalue = 0;

  // loop over number of tiles that need to be loaded
  // force ceiling so we can cover all A Columns
  for (int m = 0; m < (numAColumns + TILE_WIDTH - 1)/TILE_WIDTH; ++m) {
    // Loading of M and N tiles into shared memory
    
    // Check if tile passes right or bottom bound of A, 0 if outside
    if (m*TILE_WIDTH+tx < numAColumns && Row < numARows) {
      subTileA[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
    }
    else {
      printf("subTileA[%d,%d] not filled\n", ty, tx);
      subTileA[ty][tx] = 0.0;
    }
    // Check if tile passese bottom or right bound of B
    if (m*TILE_WIDTH+ty < numBRows && Col < numBColumns) {
      subTileB[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns + Col];
    }
    else {
      printf("subTileB[%d,%d] not filled\n", ty, tx);
      subTileB[ty][tx] = 0.0;
    }
    __syncthreads();
    
    // Do partial sum for loaded tiles once tiles are loaded
    for (int k = 0; k < TILE_WIDTH; ++k)
      Pvalue += subTileA[ty][k] * subTileB[k][tx];
    __syncthreads();
  }
  // Only load Pvals into C if item in C is in bound
  if (Row < numCRows && Col < numCColumns) {
    C[Row*numCColumns+Col] = Pvalue;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
  
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory
  cudaMalloc((void **) &deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void **) &deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void **) &deviceC, numCRows * numCColumns * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions
  dim3 DimGrid(ceil(numCColumns/(TILE_WIDTH * 1.0)), ceil(numCRows/(TILE_WIDTH * 1.0)), 1);
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH , 1);
  wbLog(TRACE, "DimGrid", ceil(numCRows/(TILE_WIDTH * 1.0)), " x ", ceil(numCColumns/(TILE_WIDTH * 1.0)));
  wbLog(TRACE, "DimBlock", TILE_WIDTH, " x ", TILE_WIDTH);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory
  cudaFree(deviceA); 
  cudaFree(deviceB); 
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
