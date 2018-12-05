#include "tensorflow/c/c_api.h"
#include <stdio.h>
#include <memory.h>
#include <assert.h>
#include <vector>
#include <iostream>

TF_Buffer* read_file(const char* file);
static void deallocator(void* data, size_t length, void* arg);
void free_buffer(void* data, size_t length);

int main() 
{
	// -------------------- read graph --------------------
	
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

	/*std::cout << "Graph opeartions:" << std::endl;
	int idx = 0;
	while(true)
	{
		size_t pos = idx;
		TF_Operation* next_op = TF_GraphNextOperation(graph, &pos);
		
		if (next_op == NULL)
			break;
		
		std::string op_name = TF_OperationName(next_op);
		std::cout << "idx: " << idx<< " operation: " << op_name << "\n";
		idx = idx + 1;
	}*/
	
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
	
	// -------------------- create session --------------------

	TF_SessionOptions* sess_opts = TF_NewSessionOptions();
	TF_Session* session = TF_NewSession(graph, sess_opts, status);
	assert(TF_GetCode(status) == TF_OK);

	// -------------------- run init operation --------------------

	TF_Operation* init_op = TF_GraphOperationByName(graph, "init_vars"); 
	std::vector<TF_Operation*> ops;
	ops.push_back(init_op);

	TF_SessionRun(session, nullptr,
				nullptr, nullptr, 0,
				nullptr, nullptr, 0,
				&ops[0], 1, nullptr, status);

	// -------------------- run main operation --------------------

	const int in_num_bytes = 4 * sizeof(float);
	int64_t in_dims[] = {4};
	
	float* in_values = (float*)malloc(in_num_bytes);
	for (int i=0; i<4; i++)
		in_values[i]=2.0f;

	std::cout << "Input vector:" << std::endl;
	for (int i = 0; i < 4; i++)
		std::cout << in_values[i] << " ";
	std::cout << std::endl;

	std::vector<TF_Output> inputs; // input operations
	TF_Operation* input_op = TF_GraphOperationByName(graph, "x"); // std::cout << "Input operation num outputs: " << TF_OperationNumOutputs(input_op) << "\n";
	TF_Output output_from_input_op = {input_op, 0};
	inputs.push_back(output_from_input_op);

	std::vector<TF_Tensor*> input_values; // input tensor values
	TF_Tensor* input = TF_NewTensor(TF_FLOAT, in_dims, 1, in_values, in_num_bytes, &deallocator, 0); // TF_Dim(input, 0)
	input_values.push_back(input);

	const int out_num_bytes = 4 * sizeof(float);
	int64_t out_dims[] = {4};

	std::vector<TF_Output> outputs; // output operations
	TF_Operation* output_op = TF_GraphOperationByName(graph, "y");
	TF_Output output_from_output_op = {output_op, 0};
	outputs.push_back(output_from_output_op);

	std::vector<TF_Tensor*> output_values; // output tensor values
	TF_Tensor* output_value = TF_AllocateTensor(TF_FLOAT, out_dims, 1, out_num_bytes);
	output_values.push_back(output_value);

	TF_SessionRun(session, nullptr,
				&inputs[0], &input_values[0], inputs.size(),
				&outputs[0], &output_values[0], outputs.size(),
				nullptr, 0, nullptr, status);

	// -------------------- collect results --------------------

	float* out_vals = (float*)(TF_TensorData(output_values[0]));

	std::cout << "Output vector:" << std::endl;
	for (int i=0; i<4; i++)
		std::cout << out_vals[i] << " ";
	std::cout << std::endl;

	// -------------------- free memory --------------------

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

