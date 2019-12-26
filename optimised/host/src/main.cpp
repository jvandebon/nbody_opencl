// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

#define STRING_BUFFER_LEN 1024

/** DEFINE RUNTIME CONSTANTS **/
#define NN 8192

typedef struct {
    float x;
    float y;
    float z;
    float padding;
} coord3d_t;

typedef struct {
    coord3d_t p;
    coord3d_t v;
} particle_t;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
/** DEFINE EACH KERNEL **/
static cl_kernel kernel_lib = NULL;
static cl_program program = NULL;
/** DEFINE EACH KERNEL ARG **/
static cl_mem m = NULL;
static cl_mem in_p = NULL;
static cl_mem a = NULL;
static cl_int N = NN;


// Function prototypes
bool init();
void cleanup();
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name);
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name);
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name);
static void device_info_string( cl_device_id device, cl_device_info param, const char* name);
static void display_device_info( cl_device_id device );

// Entry point.
int main(int argc, char** argv) {

   bool emulator_run = false;
   if ( (argc > 1) && (strcmp (argv[1], "-emulator") == 0) ) {
      emulator_run = true;
   }

  cl_int status;

  if(!init()) {
    return -1;
  }

  /** DEFINE INPUT DATA AND ENQUEUE **/
  printf("Generate input data...\n");
  particle_t *particles = (particle_t *) aocl_utils::alignedMalloc(NN*sizeof(particle_t));
  float *m_in = (float *) aocl_utils::alignedMalloc(NN*sizeof(float));;

  srand(100);
  for (int i = 0; i < NN; i++)
  {
      m_in[i] = (float)rand()/100000;
      particles[i].p.x = (float)rand()/100000;
      particles[i].p.y = (float)rand()/100000;
      particles[i].p.z = (float)rand()/100000;
      particles[i].v.x = (float)rand()/100000;
      particles[i].v.y = (float)rand()/100000;
      particles[i].v.z = (float)rand()/100000;

/*    if(i%(NN/10)==0){
        printf("%d -- %f %f %f %f %f %f %f\n", i, particles[i].p.x, particles[i].p.y, particles[i].p.z, particles[i].v.x, particles[i].v.y, particles[i].v.z, m_in[i]);
    }
*/
  }
  status = clEnqueueWriteBuffer(queue,m,0,0,NN*sizeof(float),m_in,0,0,0);
  checkError(status, "Failed to enqueue writing to input buffer");
  status = clEnqueueWriteBuffer(queue,in_p,0,0,NN*sizeof(particle_t),particles,0,0,0);
  checkError(status, "Failed to enqueue writing to input buffer");



  /** SET KERNEL ARGUMENTS **/
  status = clSetKernelArg(kernel_lib,0,sizeof(cl_int),&N) ;
  checkError(status, "Failed to set kernel_lib arg 0");
  status = clSetKernelArg(kernel_lib,1,sizeof(cl_mem),&m);
  checkError(status, "Failed to set kernel_lib arg 1");
  status = clSetKernelArg(kernel_lib,2,sizeof(cl_mem),&in_p);
  checkError(status, "Failed to set kernel_lib arg 2");
  status = clSetKernelArg(kernel_lib,3,sizeof(cl_mem),&a);
  checkError(status, "Failed to set kernel_lib arg 3");



  // Launch the kernel
  printf("Enqueueing kernel...\n");
  double lib_start = aocl_utils::getCurrentTimestamp();

  /** ENQUEUE CORRECT KERNEL NAME **/
  //status = clEnqueueTask(queue, kernel_lib, 0, NULL, NULL);
  size_t globalWorkSize = NN;
  size_t workGroupSize = NN;
  status = clEnqueueNDRangeKernel(queue, kernel_lib, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel_lib");

  status = clFinish(queue);
  checkError(status, "Failed to finish");

  double lib_stop = aocl_utils::getCurrentTimestamp();
  printf ("Kernel computation took %g seconds\n", lib_stop - lib_start);

  /** READ/HANDLE OUTPUT DATA **/
  printf("Reading results to buffers...\n");
  coord3d_t *a_out =  (coord3d_t *) aocl_utils::alignedMalloc(NN*sizeof(coord3d_t));
  status = clEnqueueReadBuffer(queue, a, 1, 0, NN*sizeof(coord3d_t), a_out, 0, 0, 0);
  checkError(status, "Failed to enqueue read buffer output to a_out");
  status = clFinish(queue);
  checkError(status, "Failed to read buffer output to a_out");
  
  particle_t *out_particles = (particle_t *) aocl_utils::alignedMalloc(NN*sizeof(particle_t));
  for (int i = 0; i < NN; i++){
      out_particles[i].p.x = particles[i].p.x + particles[i].v.x;
      out_particles[i].p.y = particles[i].p.y + particles[i].v.y;
      out_particles[i].p.z = particles[i].p.z + particles[i].v.z;
      out_particles[i].v.x = particles[i].v.x + a_out[i].x;
      out_particles[i].v.y = particles[i].v.y + a_out[i].y;
      out_particles[i].v.z = particles[i].v.z + a_out[i].z;
  }

  double stop = aocl_utils::getCurrentTimestamp();
  printf ("Toal computation took %g seconds\n", stop - lib_start);

  /** ADD OPTIONAL CODE TO VERIFY RESULTS **/
  /*printf("Checking results...\n\n");
  for (int i = 0; i < NN; i++){
      if(i%(NN/10) ==0){
          printf("%d -- %f %f %f %f %f %f\n", i, out_particles[i].p.x, out_particles[i].p.y, out_particles[i].p.z, out_particles[i].v.x, out_particles[i].v.y, out_particles[i].v.z);
      }
  }*/

  // Free the resources allocated
  cleanup();
  /** CLEAR ANY ALLOCATED RESOURCES **/
  aocl_utils::alignedFree(m_in);
  aocl_utils::alignedFree(particles);
  aocl_utils::alignedFree(a_out);
  aocl_utils::alignedFree(out_particles);
 
  return 0;
}

/////// HELPER FUNCTIONS ///////

bool init() {
  cl_int status;

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // User-visible output - Platform information
  /*({
    char char_buffer[STRING_BUFFER_LEN]; 
    printf("Querying platform for info:\n");
    printf("==========================\n");
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
  }*/

  // Query the available OpenCL devices.
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;

  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

  // We'll just use the first device.
  device = devices[0];

  // Display some device information.
//  display_device_info(device);

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the command queue.
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Create the program.
  /** SPECIFY KERNEL FILENAME **/
  std::string binary_file = getBoardBinaryFile("kernel", device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  /** SPECIFY KERNEL NAME **/
  kernel_lib = clCreateKernel(program, "kernel_lib", &status);
  checkError(status, "Failed to create kernel");
  size_t preferredGroupSize;
clGetKernelWorkGroupInfo(kernel_lib,
		device,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(size_t),
		&preferredGroupSize,
		NULL);
printf("CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: %d", preferredGroupSize);
  // Create the input and output buffers
  /** CREATE IN/OUT BUFFERS DEPENDING ON ARGS **/
  m = clCreateBuffer(context, CL_MEM_READ_ONLY, NN*sizeof(float), 0, &status);
  checkError(status, "Failed to create the input buffer");
  in_p = clCreateBuffer(context, CL_MEM_READ_ONLY, NN*sizeof(particle_t), 0, &status);
  checkError(status, "Failed to create the output buffer");
  a = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NN*sizeof(coord3d_t), 0, &status);
  checkError(status, "Failed to create the output buffer");
  return true;
}

// Free the resources allocated during initialization
void cleanup() {
  if(kernel_lib) {
    clReleaseKernel(kernel_lib);  
  }
  if(program) {
    clReleaseProgram(program);
  }
  if(queue) {
    clReleaseCommandQueue(queue);
  }
  if(context) {
    clReleaseContext(context);
  }
  /** CLEAR ALL BUFFERS **/
  if(m){
    clReleaseMemObject(m);
  }
  if(in_p){
    clReleaseMemObject(in_p);
  }
  if(a){
    clReleaseMemObject(a);
  }
}

// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name) {
  cl_ulong a;
  clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
  printf("%-40s = %lu\n", name, a);
}
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name) {
  cl_uint a;
  clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
  printf("%-40s = %u\n", name, a);
}
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name) {
  cl_bool a;
  clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
  printf("%-40s = %s\n", name, (a?"true":"false"));
}
static void device_info_string( cl_device_id device, cl_device_info param, const char* name) {
  char a[STRING_BUFFER_LEN]; 
  clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
  printf("%-40s = %s\n", name, a);
}

// Query and display OpenCL information on device and runtime environment
static void display_device_info( cl_device_id device ) {

  printf("Querying device for info:\n");
  printf("========================\n");
  device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
  device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
  device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
  device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
  device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
  device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
  device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
  device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
  device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
  device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
  device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
  device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
  device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
  device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
  device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
  device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
  device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");
  device_info_uint(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE");
{
    cl_command_queue_properties ccp;
    clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
    printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)?"true":"false"));
    printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE)?"true":"false"));
  }
}

