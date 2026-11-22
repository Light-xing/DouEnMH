/**
 * Cross-validation: load Python-generated input tensors from .bin files,
 * run inference through LibTorch .pt models, compare output with Python.
 *
 * MSVC compile:
 *   MSYS_NO_PATHCONV=1 cl /std:c++17 /EHsc /MD cross_validate.cpp ^
 *     /I<h:/bot/environments/libtorch/libtorch/include> ^
 *     /I<h:/bot/environments/libtorch/libtorch/include/torch/csrc/api/include> ^
 *     /link <h:/bot/environments/libtorch/libtorch/lib/torch.lib> ^
 *            <h:/bot/environments/libtorch/libtorch/lib/torch_cpu.lib> ^
 *            <h:/bot/environments/libtorch/libtorch/lib/c10.lib> ^
 *     /out:cross_validate.exe
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <torch/script.h>

// ---- Read .bin tensor file ----
// Format: [n_dims int32][dim0 int32]...[dimN int32][float32 data...]
torch::Tensor load_bin_tensor(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open: " + path);
    }

    // Read ndim
    int32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    if (!file) throw std::runtime_error("Failed to read ndim from: " + path);

    // Read dimensions
    std::vector<int64_t> dims(ndim);
    for (int i = 0; i < ndim; i++) {
        int32_t d;
        file.read(reinterpret_cast<char*>(&d), sizeof(d));
        dims[i] = static_cast<int64_t>(d);
    }

    // Count total elements
    int64_t numel = 1;
    for (auto d : dims) numel *= d;

    // Read float32 data
    std::vector<float> data(numel);
    file.read(reinterpret_cast<char*>(data.data()), numel * sizeof(float));
    if (!file) throw std::runtime_error("Failed to read data from: " + path);

    // Create tensor (clone to own the data)
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor t = torch::from_blob(data.data(), dims, options).clone();
    return t;
}

// ---- Compare tensors ----
bool compare_tensors(const torch::Tensor& a, const torch::Tensor& b, double rtol = 1e-4, double atol = 1e-5) {
    if (a.sizes() != b.sizes()) {
        std::cerr << "  Shape mismatch: " << a.sizes() << " vs " << b.sizes() << std::endl;
        return false;
    }
    auto diff = torch::abs(a - b);
    auto max_diff = diff.max().item<double>();
    auto mean_diff = diff.mean().item<double>();

    // Check if close: diff <= atol + rtol * abs(b)
    auto allowed = atol + rtol * torch::abs(b);
    auto close = diff <= allowed;
    bool all_close = close.all().item<bool>();

    std::cout << "  Max diff: " << max_diff << "  Mean diff: " << mean_diff << std::endl;
    if (!all_close) {
        auto num_diff = torch::sum(torch::logical_not(close)).item<int64_t>();
        std::cout << "  [FAIL] " << num_diff << " / " << a.numel()
                  << " elements differ beyond tolerance (rtol=" << rtol << " atol=" << atol << ")" << std::endl;
        // Print first few differing values
        auto flat_a = a.flatten();
        auto flat_b = b.flatten();
        auto flat_diff = diff.flatten();
        int printed = 0;
        for (int64_t i = 0; i < flat_a.numel() && printed < 5; i++) {
            double da = flat_a[i].item<double>();
            double db = flat_b[i].item<double>();
            double d = flat_diff[i].item<double>();
            double al = atol + rtol * std::abs(db);
            if (d > al) {
                std::cout << "    [" << i << "] cpp=" << da << "  py=" << db << "  diff=" << d << std::endl;
                printed++;
            }
        }
        return false;
    }
    std::cout << "  [OK] All " << a.numel() << " elements match within tolerance" << std::endl;
    return true;
}

// ---- Test one model ----
int test_model(const std::string& data_dir, const std::string& name,
               const std::string& model_dir, bool is_bid) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "Testing: " << name << (is_bid ? " (bid)" : " (play)") << std::endl;

    try {
        // Load model
        std::string model_path = model_dir + "/" + name + "_jit.pt";
        std::cout << "Loading model: " << model_path << std::endl;
        torch::jit::Module model = torch::jit::load(model_path);
        model.eval();

        // Load input tensors
        std::string in1_path, in2_path;
        if (is_bid) {
            in1_path = data_dir + "/" + name + "_z.bin";
            in2_path = data_dir + "/" + name + "_x.bin";
        } else {
            in1_path = data_dir + "/" + name + "_src1.bin";
            in2_path = data_dir + "/" + name + "_src2.bin";
        }

        auto input1 = load_bin_tensor(in1_path);
        auto input2 = load_bin_tensor(in2_path);
        std::cout << "Input1: " << input1.sizes() << "  Input2: " << input2.sizes() << std::endl;

        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input1);
        inputs.push_back(input2);
        auto output = model.forward(inputs).toTensor();
        std::cout << "Output: " << output.sizes() << std::endl;
        std::cout << "Q values: [";
        auto flat = output.flatten();
        for (int64_t i = 0; i < flat.size(0); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << flat[i].item<float>();
        }
        std::cout << "]" << std::endl;
        int64_t best = output.argmax().item<int64_t>();
        std::cout << "Best action (argmax): " << best << std::endl;

        // Load Python output for comparison
        std::string out_path = data_dir + "/" + name + "_out.bin";
        auto py_output = load_bin_tensor(out_path);

        // Compare
        std::cout << "Comparing with Python output:" << std::endl;
        bool ok = compare_tensors(output, py_output);

        // Also check that the model discriminates (different Q for different actions)
        float q0 = flat[0].item<float>();
        bool discriminates = false;
        for (int64_t i = 1; i < flat.size(0); i++) {
            if (std::abs(flat[i].item<float>() - q0) > 1e-6f) {
                discriminates = true;
                break;
            }
        }
        std::cout << "Model discriminates actions: " << (discriminates ? "YES" : "NO") << std::endl;

        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "  [FAIL] Exception: " << e.what() << std::endl;
        return 1;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <data_dir> <model_dir>" << std::endl;
        std::cerr << "Example: " << argv[0] << " cross_validate/data h:/bot/DouEnMix-design/bridge/export_model/<suffix>" << std::endl;
        return 1;
    }

    std::string data_dir = argv[1];
    std::string model_dir = argv[2];

    int failures = 0;

    // Bid models
    failures += test_model(data_dir, "first",  model_dir, true);
    failures += test_model(data_dir, "second", model_dir, true);
    failures += test_model(data_dir, "third",  model_dir, true);

    // Play models
    failures += test_model(data_dir, "landlord",      model_dir, false);
    failures += test_model(data_dir, "landlord_up",   model_dir, false);
    failures += test_model(data_dir, "landlord_down", model_dir, false);

    std::cout << "\n============================================================" << std::endl;
    if (failures == 0) {
        std::cout << "All 6 models passed cross-validation!" << std::endl;
    } else {
        std::cout << failures << " / 6 models FAILED cross-validation" << std::endl;
    }
    std::cout << "============================================================" << std::endl;

    return failures > 0 ? 1 : 0;
}
