#include "tensorflow/c/c_api.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <algorithm>

TF_Buffer* read_file(const char* file);
static void deallocator(void* data, size_t length, void* arg);
void free_buffer(void* data, size_t length);

void convert_data_to_mat (std::vector<cv::Mat> &imgs, int64_t* samples_dims, int samples_num_dims, float* samples_data)
{
	assert(samples_num_dims == 4);
	
	int batch_size = samples_dims[0];
	int sample_num_pixels = 1;
	for (int i=1; i<4; i++)
	{
		sample_num_pixels = sample_num_pixels * samples_dims[i];
	}
	
	for (int i=0; i<batch_size; i++)
	{
		cv::Mat img(samples_dims[1], samples_dims[2], CV_32FC3);

		// both opencv and tensorflow store image in row -> column -> depth major order so cv::Mat data pointer can be overwritten
		img.data = (uchar*)(samples_data + i * sample_num_pixels);
		
		// each float pixel value from range [-1.0, 1.0] map to uchar pixel value from range [0, 255] -> y = (uchar*)(127.0 * x + 128.0)
		img.convertTo(img, CV_8UC3, 127.0, 128.0);
		cvtColor(img, img, CV_RGB2BGR);
		imgs.push_back(img);
	}
}

void process_classes_names(std::string filepath, std::string* classes)
{
	std::ifstream file(filepath);

	if (file.is_open()) 
	{
		std::string line;
		while (std::getline(file, line))
		{
			int delimiter = line.find(':');
			int idx = std::stoi(line.substr(1, delimiter - 1));
			std::string name = line.substr(delimiter + 3, line.length() - 9);
			
			std::transform(name.begin(), name.end(), name.begin(), ::tolower);
			classes[idx] = name;
		}
		file.close();
	}
}

int main() 
{
	// -------------------------- read graph ---------------------------
	
	TF_Buffer* graph_def = read_file("/tmp/python/model/graph.pb");
	TF_Graph* graph = TF_NewGraph();

	TF_Status* status = TF_NewStatus();
	TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
	TF_GraphImportGraphDef(graph, graph_def, graph_opts, status);
	
	if (TF_GetCode(status) != TF_OK) 
	{
		  std::cout<< "Error while importing graph: " << TF_Message(status) << std::endl;
		  return 1;
	}
	else 
	{
		  std::cout << "Graph imported." << std::endl;
	}

	/*void TF_SessionRun(
		TF_Session* session,
		// RunOptions
		const TF_Buffer* run_options,
		// Input tensors
		const TF_Output* inputs, TF_Tensor* const* input_values, int ninputs,
		// Output tensors
		const TF_Output* outputs, TF_Tensor** output_values, int noutputs,
		// Target operations
		const TF_Operation* const* target_opers, int ntargets,
		// RunMetadata
		TF_Buffer* run_metadata,
		// Output status
		TF_Status*);*/
	
	// ------------------------ create session -------------------------

	TF_SessionOptions* sess_opts = TF_NewSessionOptions();
	TF_Session* session = TF_NewSession(graph, sess_opts, status);
	assert(TF_GetCode(status) == TF_OK);

	// ---------------------- run init operation -----------------------

	TF_Operation* init_op = TF_GraphOperationByName(graph, "init_vars"); 
	std::vector<TF_Operation*> ops;
	ops.push_back(init_op);

	TF_SessionRun(session, nullptr,
				nullptr, nullptr, 0,
				nullptr, nullptr, 0,
				&ops[0], 1, nullptr, status);

	// ---------------------- run main operation -----------------------

	// Two output tensors: 
	// 1. indexes name="random_uniform:0" shape=(2,) dtype=int32
	// 2. samples name="module_apply_default/G_trunc_output:0" shape=(2, 512, 512, 3) dtype=float32
	
	std::vector<TF_Output> outputs; // output operations
	TF_Operation* samples_op = TF_GraphOperationByName(graph, "module_apply_default/G_trunc_output");
	TF_Output samples_op_output = {samples_op, 0};	
	outputs.push_back(samples_op_output);	
	TF_Operation* indexes_op = TF_GraphOperationByName(graph, "random_uniform");
	TF_Output indexes_op_output = {indexes_op, 0};
	outputs.push_back(indexes_op_output);

	std::vector<TF_Tensor*> output_tensors; // output tensor values
	int samples_num_dims = TF_GraphGetTensorNumDims(graph,samples_op_output, status);
	int64_t* samples_dims = (int64_t*)malloc(samples_num_dims * sizeof(int64_t));
	TF_GraphGetTensorShape(graph, samples_op_output, samples_dims, samples_num_dims, status);
	int samples_num_bytes = sizeof(float);
	for (int i=0; i<samples_num_dims; i++)
		samples_num_bytes = samples_num_bytes * samples_dims[i];
		
	TF_Tensor* samples_tensor = TF_AllocateTensor(TF_FLOAT, samples_dims, samples_num_dims, samples_num_bytes);
	output_tensors.push_back(samples_tensor);
	
	int batch_size = samples_dims[0];
	const int indexes_num_bytes = batch_size * sizeof(int);
	int64_t indexes_dims[] = {batch_size};		
	TF_Tensor* indexes_tensor = TF_AllocateTensor(TF_FLOAT, indexes_dims, 1, indexes_num_bytes);
	output_tensors.push_back(indexes_tensor);

	TF_SessionRun(session, nullptr,
				nullptr, nullptr, 0,
				&outputs[0], &output_tensors[0], outputs.size(),
				nullptr, 0, nullptr, status);

	// ------------------------ collect results ------------------------
					
	float* samples_data = (float*)(TF_TensorData(output_tensors[0]));
	int* indexes_data = (int*)(TF_TensorData(output_tensors[1]));
		
	std::string classes[1000];
	process_classes_names("/tmp/python/imagenet_names.txt", classes);

	std::vector<cv::Mat> imgs;
	convert_data_to_mat (imgs, samples_dims, samples_num_dims, samples_data);
	
	assert (imgs.size() == batch_size);
	std::cout << "Images generated:" << std::endl;	
	for (int i=0; i<batch_size; i++)
	{
		cv::Mat img = imgs[i];
		int idx = indexes_data[i];
		std::string name = classes[idx];
		
		cv::imwrite( "imgs/" + name + ".png", img );
		std::cout<< idx << " -> " << name << std::endl;
	}

	// -------------------------- free memory --------------------------

	TF_CloseSession(session, status);
	TF_DeleteSession(session, status);
	TF_DeleteSessionOptions(sess_opts);
	TF_DeleteImportGraphDefOptions(graph_opts);
	TF_DeleteGraph(graph);
	TF_DeleteStatus(status);
	
	return 0;		
}

void free_buffer(void* data, size_t length) 
{
	free(data);
}
static void deallocator(void* data, size_t length, void* arg) 
{
	free(data);
}

TF_Buffer* read_file(const char* file) 
{
	FILE *f = fopen(file, "rb");
	fseek(f, 0, SEEK_END);
	long fsize = ftell(f);
	fseek(f, 0, SEEK_SET);

	void* data = malloc(fsize);
	fread(data, fsize, 1, f);
	fclose(f);

	TF_Buffer* buf = TF_NewBuffer();
	buf->data = data;
	buf->length = fsize;
	buf->data_deallocator = free_buffer;
	return buf;
}
