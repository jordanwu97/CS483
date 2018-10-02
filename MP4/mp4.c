#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants
#define MASK_WIDTH 3
#define MASK_RADIUS 1

#define TILE_WIDTH 8
  
//@@ Define constant memory for device kernel
__constant__ float deviceKernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

// Kernel Code
__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  // Some constants
  int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z; 
  int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
  int _x = bx * blockDim.x + tx; 
  int _y = by * blockDim.y + ty; 
  int _z = bz * blockDim.z + tz;
  
  // Allocate shared Tile. Using strategy 3 where we load tile covering input
  // and do global memory access on halos. 
  __shared__ float inputTile[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];
  
  // Load Tile if in bound of input matrix, else load 0
  if (_x < x_size && _y < y_size && _z < z_size) {
    inputTile[tz][ty][tx] = input[_x + (x_size * _y) + (x_size * y_size * _z)];
  } else {
    inputTile[tz][ty][tx] = 0.0;
  }
  __syncthreads();
  
  // Do convolution
  float p = 0.0;
  for (int k = -MASK_RADIUS; k <= MASK_RADIUS; k++) {
    for (int j = -MASK_RADIUS; j <= MASK_RADIUS; j++) {
      for (int i = -MASK_RADIUS; i <= MASK_RADIUS; i++) {
        int txi = tx + i;
        int tyj = ty + j;
        int tzk = tz + k;
        
        int im = i+MASK_RADIUS;
        int jm = j+MASK_RADIUS;
        int km = k+MASK_RADIUS;
        
        // if halos of tile, replicate inefficient algorithm. Go to global mem to retrieve element.
        if (txi < 0 || txi >= TILE_WIDTH || tyj < 0 || tyj >= TILE_WIDTH || tzk < 0 || tzk >= TILE_WIDTH) {
          if (_x+i >= 0 && _y+j >= 0 && _z+k >=0 && _x+i < x_size && _y+j < y_size && _z+k < z_size) {
            int im = i+MASK_RADIUS;
            int jm = j+MASK_RADIUS;
            int km = k+MASK_RADIUS;
            p += input[(_x+i) + (x_size * (_y+j)) + (x_size * y_size * (_z+k))] * deviceKernel[km][jm][im];
          }
        }
        // if inside tile, all gucci. perform convolution with cached tile
        else {
          p += inputTile[tzk][tyj][txi] * deviceKernel[km][jm][im];
        }
      }
    }
  }
  // save output if in bound
  if (_x < x_size && _y < y_size && _z < z_size) {
    output[_x + (x_size * _y) + (x_size * y_size * _z)] = p;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory
  cudaMalloc((void **) &deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength - 3) * sizeof(float));
  
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU
  cudaMemcpy(deviceInput, &hostInput[3], (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength * sizeof(float));
  
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions
  dim3 DimGrid(ceil(x_size/(TILE_WIDTH * 1.0)), ceil(y_size/(TILE_WIDTH * 1.0)), ceil(z_size/(TILE_WIDTH * 1.0)));
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
  
  //@@ Launch the GPU kernel
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");
  
  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host
  cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}