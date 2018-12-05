#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

namespace tf = tensorflow;

tf::Tensor mat2tensor(cv::Mat img)
{
	cv::resize(img, img, cv::Size(224, 224));
    cv::cvtColor(img, img, CV_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0/255.0);
	
	tf::Tensor t = tf::Tensor(tf::DT_FLOAT, tf::TensorShape({1, 224, 224, 3}));

	// there is no direct access to Tensor buffer in C++ interface, so it's impossible to simply pass pointer	
	// both opencv and tensorflow store image in row -> column -> depth major order so data can be copied
	tf::StringPiece t_data = t.tensor_data();
	memcpy(const_cast<char*>(t_data.data()), (img.data), 224 * 224 * 3 * sizeof(float));

	// auto t_mapped = t.tensor<float, 4>();
	// float* img_pointer = (float*)img.data;
	
	return t;
}

void process_classes_names(std::string filepath, std::string* classes)
{
	std::ifstream file(filepath);
	classes[0] = "none";

	if (file.is_open()) 
	{
		std::string line;
		while (std::getline(file, line))
		{
			int delimiter = line.find(':');
			int idx = std::stoi(line.substr(1, delimiter-1));
			std::string name = line.substr(delimiter + 3, line.length() - 9);

			std::transform(name.begin(), name.end(), name.begin(), ::tolower);
			classes[idx+1]=name;
		}
		file.close();
	}
}

int main(int argc, char* argv[]) 
{
	cv::Mat img = cv::imread("/tmp/python/tiger.jpg");
	tf::Tensor t = mat2tensor(img);
	
	// initialize tf session
	tf::Session* session;
	tf::Status status = tf::NewSession(tf::SessionOptions(), &session);
	if (!status.ok())
	{std::cout << status.ToString() << "\n"; return 1;}
	
	// read protobuf graph exported using tf.train.write_graph
	tf::GraphDef graph_def;
	status = tf::ReadBinaryProto(tf::Env::Default(), "/tmp/python/model/graph.pb", &graph_def);
	if (!status.ok()) 
	{std::cout << status.ToString() << "\n"; return 1;}
	
	// add graph to the session
	status = session->Create(graph_def);
	if (!status.ok()) 
	{std::cout << status.ToString() << "\n"; return 1;}
	
	// pass tensor to placeholder (any tensor in graph can be overwritten here)
    std::vector<std::pair<std::string, tf::Tensor>> inputs = {{ "x", t }};
    std::vector<tf::Tensor> outputs;

	// run the session
	status = session->Run(inputs, {}, {"init_vars"}, &outputs);
	status = session->Run(inputs, {"predictions:1", "probabilities:0"}, {}, &outputs);
	
	std::string classes[1001];
	process_classes_names("/tmp/python/imagenet_names.txt", classes);
	
	tf::Tensor top_k_tensor = outputs[0];
	tf::Tensor probs_tensor = outputs[1];
	
    std::cout << "Top 5 predicitions:" << std::endl;
    for(int i=0; i<5; i++)
    {
		int idx = top_k_tensor.tensor<int, 2>()(0, i);
				
		std::cout << "--class: ";
		std::cout << classes[idx];
		std::cout << " --probability: ";
		std::cout << probs_tensor.tensor<float, 2>()(0, idx) << std::endl;;
	}

	return 0;
}
