#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

using namespace tensorflow;

int main(int argc, char* argv[]) {
	
	// initialize tf session
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok())
	{std::cout << status.ToString() << "\n"; return 1;}

	// read protobuf graph exported using tf.train.write_graph
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), "/tmp/python/model/graph.pb", &graph_def);
	if (!status.ok()) 
	{std::cout << status.ToString() << "\n"; return 1;}

	// add graph to the session
	status = session->Create(graph_def);
	if (!status.ok()) 
	{std::cout << status.ToString() << "\n"; return 1;}
	
	// create tensor for placeholder
	Tensor t = Tensor(DT_FLOAT,TensorShape({4}));
	auto t_mapped = t.tensor<float, 1>(); //auto t_vec = t.vec<float>();
	for (int i=0; i<4; i++)
		t_mapped(i) = 2.0;
	
	std::cout << "Input vector:" << "\n";
	for (int i=0; i<4; i++)
		std::cout << t.vec<float>()(i) << " ";
	std::cout << "\n";

	// pass tensor to placeholder (any tensor in graph can be overwritten here)
    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {{ "x", t }};
    std::vector<tensorflow::Tensor> outputs;

	// run the session
	status = session->Run(inputs, {}, {"init_vars"}, &outputs);
	status = session->Run(inputs, {"y"}, {}, &outputs);
	if (!status.ok()) 
	{std::cout << status.ToString() << "\n"; return 1;}

	std::cout << "Output vector:" << "\n";
	for (int i=0; i<4; i++)
		std::cout << outputs[0].vec<float>()(i) << " ";
	std::cout << "\n";
	
	session->Close();
	return 0;
}
