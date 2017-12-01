#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <thread>

#include <CL/cl.hpp>

#define SIZE 200


using namespace std;

void FloidMethod(int** , int** , int );
void ParallelItem(int**, int**, int, int, int);
void Paralel(int **, int**, int, int);
void WithOpenCL(int**, int**);

int main(void)
{
		int** res = new  int*[SIZE];
		int** matrix = new int*[SIZE];
		for (int i = 0; i < SIZE; ++i)
		{
			matrix[i] = new int[SIZE];
			res[i] = new int[SIZE];
		}
		
		srand(time(0));
		for (std::size_t i = 0; i < SIZE; ++i)
		{
			for (std::size_t j = 0; j < SIZE; ++j)
			{
				if (i == j)
				{
					matrix[i][j] = 0;
				}
				else
				{
					matrix[i][j] = (rand() % 100 + 1);
				}
			}
		}
		cout << "Floid Algorithm" << endl;
	
		clock_t start_point = clock();
	
		FloidMethod(matrix, res, SIZE);
	
		clock_t end_point = clock();
		cout << "\nSingle thread: " << ((double)(end_point - start_point) / CLK_TCK) << endl;
		cout <<" res[2][5] = "<< res[2][5] << endl;
	
	//////////////////////////////////////////////////////////////////////////////
		
		start_point = clock();
	
		Paralel(matrix,res, SIZE,4);
	
		end_point = clock();
		cout << "\nFour threads: " << ((double)(end_point - start_point) / CLK_TCK) << endl;
		cout <<" res[2][5] = "<< res[2][5] << endl;
	
	//////////////////////////////////////////////////////////////////////////////
	
		WithOpenCL(matrix, res);
		cout << " res[2][5] = " << res[2][5] << endl;

	int a;
	cin >> a;

	return 0;
}


void FloidMethod(int** matrix, int** D, int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			D[i][j] = matrix[i][j];
		}
	}

	for (int i = 0; i < size; i++)
	{
		D[i][i] = 0;
	}

	for (int m = 0; m < size; m++)
	{
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				if ((D[i][j] >(D[i][m] + D[m][j]) || D[i][j] == -1)
					&& D[i][m] != -1
					&& D[m][j] != -1)
				{
					D[i][j] = (D[i][m] + D[m][j]);
				}

			}
		}
	}

}

void ParallelItem(int** matrix, int** D, int start, int end, int size)
{
	for (int m = start; m < end; m++)
	{
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				if (D[i][j] >(D[i][m] + D[m][j]) && D[i][j] != 0 && D[i][m] != 0 && D[m][j] != 0)
				{
					D[i][j] = (D[i][m] + D[m][j]);
				}

			}
		}
	}
}
void Paralel(int ** matrix, int** D, int size, int count_threads)
{
	thread* threads = new thread[count_threads];

	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			if (i == j)
				D[i][j] = 0;
			else
				D[i][j] = matrix[i][j];
		}
	}

	int start_point = 0;
	int end_point = size / count_threads;
	for (int j = 0; j < count_threads; ++j)
	{
		if (j == count_threads - 1)
		{
			int end = end_point + (size % count_threads);
			threads[j] = thread(ParallelItem, matrix, D, start_point, end, size);
		}
		else
		{
			threads[j] = thread(ParallelItem, matrix, D, start_point, end_point, size);
		}

		start_point += (size / count_threads);
		end_point += (size / count_threads);
	}

	for (int i = 0; i < count_threads; ++i)
	{
		threads[i].join();
	}
}

void WithOpenCL(int** matrix, int** res)
{

	int path_dis_matrix[SIZE*SIZE];
	int path_matrix[SIZE*SIZE];
	int p = 0;
	for (int i = 0; i<SIZE; i++) {
		for (int j = 0; j < SIZE; j++)
		{
			path_dis_matrix[p++] = matrix[i][j];
		}
	}


	for (cl_int i = 0; i < SIZE; ++i)
	{
		for (cl_int j = 0; j < i; ++j)
		{
			path_matrix[i * SIZE + j] = i;
			path_matrix[j * SIZE + i] = j;
		}
		path_matrix[i * SIZE + i] = i;
	}


	clock_t start_point, end_point;
	cl_context context;
	cl_context_properties properties[3];
	cl_uint num_of_devices = 0;
	cl_platform_id platform_id;
	cl_mem dis_buffer, buffer;
	cl_kernel kernel;
	cl_command_queue cmd_queue;
	cl_program prog;
	cl_int er;
	cl_device_id device_id;
	cl_uint count_of_platforms = 0;

	size_t global[2];
	size_t local[2];

	const char *kernelCode =
		"__kernel void floid_method(__global int * distBuffer, __global int * pathBuffer, const int numNodes, const int st) \n"\
		"{ \n"\
		"int x = get_global_id(0); \n"\
		"int y = get_global_id(1); \n"\
		"int oldWeight = distBuffer[y * numNodes + x]; \n"\
		"int tempWeight = (distBuffer[y * numNodes + st] + distBuffer[st * numNodes + x]); \n"\
		"if (tempWeight < oldWeight){ \n"\
		"distBuffer[y * numNodes + x] = tempWeight; \n"\
		" } \n"\
		"} \n"\
		"\n";

	if (clGetPlatformIDs(1, &platform_id, &count_of_platforms) != CL_SUCCESS)
	{
		cout << ("Unable to get platform id\n") << endl;
		exit(1);
	}


	if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &num_of_devices) != CL_SUCCESS)
	{
		cout << ("Unable to get device_id\n") << endl;
		exit(1);
	}

	properties[0] = CL_CONTEXT_PLATFORM;
	properties[1] = (cl_context_properties)platform_id;
	properties[2] = 0;

	context = clCreateContext(properties, 1, &device_id, NULL, NULL, &er);

	cmd_queue = clCreateCommandQueue(context, device_id, 0, &er);

	prog = clCreateProgramWithSource(context, 1, (const char **)&kernelCode, NULL, &er);

	if (clBuildProgram(prog, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
	{
		printf("Error building program\n");
		exit(1);
	}

	kernel = clCreateKernel(prog, "floid_method", &er);

	dis_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * SIZE * SIZE, NULL, NULL);
	buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * SIZE * SIZE, NULL, NULL);

	clEnqueueWriteBuffer(cmd_queue, dis_buffer, CL_TRUE, 0, sizeof(int) * SIZE * SIZE, path_dis_matrix, 0, NULL, NULL);
	clEnqueueWriteBuffer(cmd_queue, buffer, CL_TRUE, 0, sizeof(int) * SIZE * SIZE, path_matrix, 0, NULL, NULL);


	int size = SIZE;
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &dis_buffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer);
	clSetKernelArg(kernel, 2, sizeof(int), &size);
	clSetKernelArg(kernel, 3, sizeof(int), &size);

	global[0] = SIZE;
	global[1] = SIZE;

	local[0] = 4;
	local[1] = 4;

	start_point = clock();
	for (int i = 0; i<SIZE; i++)
	{
		clSetKernelArg(kernel, 3, sizeof(int), &i);
		clEnqueueNDRangeKernel(cmd_queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
		clFlush(cmd_queue);
	}
	clFinish(cmd_queue);
	end_point = clock();
	clEnqueueReadBuffer(cmd_queue, dis_buffer, CL_TRUE, 0, sizeof(int) *SIZE * SIZE, path_dis_matrix, 0, NULL, NULL);
	clEnqueueReadBuffer(cmd_queue, buffer, CL_TRUE, 0, sizeof(int) * SIZE * SIZE, path_matrix, 0, NULL, NULL);



	for (int i = 0; i<SIZE; i++)
	{
		for (int j = 0; j<SIZE; j++) {
			res[i][j] = path_dis_matrix[i*SIZE + j];
		}
	}

	cout << "\nWith OpenCL :" << ((double)(end_point - start_point)) / CLK_TCK << endl;

	clReleaseMemObject(dis_buffer);
	clReleaseMemObject(buffer);
	clReleaseProgram(prog);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);
}