#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
using namespace std; //max_element(first, last) trỏ về phần tử lớn nhất
vector<float> softmax(vector<float>& guess) {
	vector<float> exps(guess.size());
	float max_guess = *max_element(guess.begin(), guess.end());
	float sum = 0.0;
	for (int i = 0; i < guess.size(); i++) {
		exps[i] = exp(guess[i] - max_guess);
		sum += exps[i];
	}
	for (int i = 0; i < guess.size(); i++) {
		exps[i] /= sum;
	}
	return exps;
}
int main() {
	//tạo môi trường
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MNIST_inference");
	Ort::SessionOptions session_options; //tạo cấu hình (cấu hình do máy tự điều chỉnh sẵn)
	string model_path = "/mnt/c/MNIST_Classification/models/mnist_model.onnx";// lấy 1 xâu con trỏ .c_str() 
	Ort::Session session(env,model_path.c_str(), session_options);// thực hiện phiên chạy // 
	//lấy input output
	Ort::AllocatorWithDefaultOptions allocator;
	auto input_name = session.GetInputNameAllocated(0, allocator);
	auto output_name = session.GetOutputNameAllocated(0, allocator);
	cout << "Input: " << input_name.get() << "\n";
    cout << "Output: " << output_name.get() << "\n";
	// Đọc ảnh thật bằng OpenCV
	cv::Mat img = cv::imread("/mnt/c/MNIST_Classification/seven.png", cv::IMREAD_GRAYSCALE); //file name + cách đọc ảnh // chỉ đọc ảnh màu xám

	if (img.empty()) {
		cerr << "Không đọc được ảnh!\n";
		return -1;
	}
	cv::resize(img, img, cv::Size(28, 28)); // ghi đè đầu ra, ghi đầu vào
	img.convertTo(img, CV_32F, 1.0 / 255.0); // chuẩn hóa [0,1]// lúc đầu pixel là số nguyên dạng 8bit nên p chuyển sang float rồi normalize về đoạn 0,1 để chạy trong neural netwwork

	// Flatten thành vector<float>
	vector<float> input_tensor_value(img.begin<float>(), img.end<float>());

	vector<int64_t> input_shape = { 1, 1, 28, 28 };
	int input_tensor_size = 1 * 1 * 28 * 28;

	//
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); //OrtDeviceAllocator
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memory_info,       // thông tin vùng nhớ
	input_tensor_value.data(),  //con trỏ tới dữ liệu
	input_tensor_size,   //số phần tử
	input_shape.data(),  // con troe tới shape
	input_shape.size()  // số chiều
	);
	const char* input_names[] = { input_name.get() };// lấy ctype cho onnxrumtime API
	const char* output_names[] = { output_name.get() };
	auto output_tensors = session.Run(
		Ort::RunOptions{ nullptr },
		input_names, &input_tensor, 1,
		output_names, 1
	);
	// Lấy kết quả
	float* logits = output_tensors.front().GetTensorMutableData<float>(); // lấy cái mảng chứa các dự đoán
	int num_classes = 10; // MNIST có 10 lớp
	vector<float> output_logits(logits, logits + num_classes);// lấy các output thô ban đầu

	// -------------------------
	// 4. Postprocessing
	// -------------------------
	auto probs = softmax(output_logits);
	auto pred_class = distance(
		probs.begin(),
		max_element(probs.begin(), probs.end())
	);

	// Ground truth (ví dụ giả lập = 8)
	int ground_truth = 8;
	cout << "Ground truth label: " << ground_truth << endl;
	cout << "Predicted class: " << pred_class << endl;

	return 0;
}