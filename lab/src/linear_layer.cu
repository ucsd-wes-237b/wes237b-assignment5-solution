#include <iostream>
#include <random>
#include <assert.h>

#include "linear_layer.h"
#include "nn_exception.h"

using namespace std;
using namespace cv;

__global__ void linearLayerForward(float *W, float* input, float* output, float* b,
									const int W_rows, const int W_cols,
									const int input_rows, const int input_cols) 
{
    //TODO: complete the linear layer forward propagation
}

__global__ void linearLayerBackprop(float *W, float* eB, float* eA,
									const int W_rows, const int W_cols,
									const int eB_rows, const int eB_cols) 
{
    //TODO: complete the linear layer backpropagation
}

__global__ void linearLayerUpdateWeights(float *eB, float* input, float* W,
									const int eB_rows, const int eB_cols,
									const int input_rows, const int input_cols, float learning_rate)
{
    //TODO: complete the gradient descent for weight updates
}

__global__ void linearLayerUpdateBias(float *eB, float* b,
									const int eB_rows, const int eB_cols,
									const int b_rows, float learning_rate)
{
    //TODO: complete the gradient descent for bias updates
}

LinearLayer::LinearLayer(string name, Shape W_shape) 
{
	W_shape.transpose();
	
	Matrix weights(W_shape);
	Matrix bias(W_shape.rows, 1);

	this->W = weights;
	this->b = bias;

	this->name = name;
	b.allocateMemory();
	W.allocateMemory();
	initializeBiasWithZeros();
	initializeWeightsRandomly();
}

LinearLayer::~LinearLayer() {}

void LinearLayer::initializeWeightsRandomly() 
{
	
	float mean = 0.0;
	float stddev = 1.0;

	theRNG().state = time(NULL);
	randn(W.data_host, Scalar(mean), Scalar(stddev));

	W.copyHostToDevice();
}

void LinearLayer::initializeWeightsHalf() 
{
	W.data_host = Scalar(0.5f);

	W.copyHostToDevice();
}


void LinearLayer::initializeBiasWithZeros()
{
	
	b.data_host = Scalar(0.0f);

	b.copyHostToDevice();
}

Matrix& LinearLayer::forward(Matrix& input)
{
//	printf("W shape : (%lu %lu)\n", W.shape.rows, W.shape.cols);
//	printf("input shape : (%lu %lu)\n", input.shape.rows, input.shape.cols);

	assert(W.shape.cols == input.shape.rows);
	
	this->input = input;

	Shape output_shape(W.shape.rows, input.shape.cols);

	output.allocateMemoryIfNotAllocated(output_shape);

	computeAndStoreLayerOutput(input);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform linear layer forward propagation");

	return output;
}

void LinearLayer::computeAndStoreLayerOutput(Matrix& input) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(output.shape.cols + block_size.x - 1) / block_size.x,
						(output.shape.rows + block_size.y - 1) / block_size.y);

	linearLayerForward<<<num_of_blocks, block_size>>>( W.data_device,
															input.data_device,
															output.data_device,
															b.data_device,
															W.shape.rows, W.shape.cols,
															input.shape.rows, input.shape.cols);
}

Matrix& LinearLayer::backprop(Matrix& eB, float learning_rate)
{
	eA.allocateMemoryIfNotAllocated(input.shape);

	computeAndStoreBackpropError(eB);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform back propagation.");

	updateBias(eB, learning_rate);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform bias update.");

	updateWeights(eB, learning_rate);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform weights update.");

	return eA;
}


void LinearLayer::computeAndStoreBackpropError(Matrix& eB) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(input.shape.cols + block_size.x - 1) / block_size.x,
						(input.shape.rows + block_size.y - 1) / block_size.y);

	linearLayerBackprop<<<num_of_blocks, block_size>>>( W.data_device,
															eB.data_device,
															eA.data_device,
															W.shape.rows, W.shape.cols,
															eB.shape.rows, eB.shape.cols);
}

void LinearLayer::updateWeights(Matrix& eB, float learning_rate) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(W.shape.cols + block_size.x - 1) / block_size.x,
						(W.shape.rows + block_size.y - 1) / block_size.y);

	linearLayerUpdateWeights<<<num_of_blocks, block_size>>>(eB.data_device,
															input.data_device,
															W.data_device,
															eB.shape.rows, eB.shape.cols,
															input.shape.rows, input.shape.cols,
															learning_rate);
}

void LinearLayer::updateBias(Matrix& eB, float learning_rate) {
	dim3 block_size(256);
	dim3 num_of_blocks( (eB.shape.rows * eB.shape.cols + block_size.x - 1) / block_size.x);

	linearLayerUpdateBias<<<num_of_blocks, block_size>>>(eB.data_device,
															b.data_device,
															eB.shape.rows, eB.shape.cols,
															b.shape.rows, learning_rate);
}

int LinearLayer::getXDim() const {
	return W.shape.cols;
}

int LinearLayer::getYDim() const {
	return W.shape.rows;
}

Matrix LinearLayer::getWeightsMatrix() const {
	return W;
}

Matrix LinearLayer::getBiasVector() const {
	return b;
}
