// =============================================================================
// model_inference.h — LibTorch 模型推理引擎 (DouEnMH V2)
// =============================================================================
// 职责:
//   1. 动态加载/卸载 TorchScript .pt 模型 (LibTorch 1.5.0)
//   2. 自动检测并转换 FP16 → FP32 (LibTorch 1.5.0 CPU 端不支持 FP16 张量运算)
//   3. 分 chunk 批量推理 (Bid: chunk≤16, Play: chunk≤50), 控制单次 forward 峰值内存
//   4. 返回 argmax Q 值动作索引 (argmax 在 C++ 端执行, 模型只输出 Q 值向量)
//
// Q 值公式 (6 头输出):
//   Q = ŵ · LPW + (1 - ŵ) · LPL
//   ŵ = (tanh(LWR) + 1) / 2
//   其中 LPW = PWin head, LPL = PLoss head, LWR = WinRate head (权重门)
//   LOR_a / LOR_p / LOR_d 为纯辅助头, 不参与 Q 值计算
//
// 输入维度:
//   叫分 Bid:  z = (B, 2, 120), x = (B, 3)
//   出牌 Play (V1 DouMH 兼容格式):
//     z_batch = (B, 41, 120), x_batch = (B, 52) or (B, 56)
//     41 rows = 1 action + 5 static + 35 history
//   出牌 Play (V2 Cross-Attention 格式, 预留):
//     state_z = (1, 5, 120), action_z = (B, 1, 120),
//     history_z = (1, 35, 120), x_batch = (B, 52/56)
// =============================================================================

#ifndef MODEL_INFERENCE_H
#define MODEL_INFERENCE_H

#include <torch/script.h>
#include <torch/torch.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>

// V2 维度常量 (与 card_encoding.h / feature_encoder.h 对齐)
constexpr int MI_Z_ROWS      = 2;    // Bid: action broadcast + hand
constexpr int MI_Z_FEAT      = 120;  // cards2array_v2 维度
constexpr int MI_ZB_ROWS     = 41;   // Play z_batch: 1 action + 5 static + 35 history
constexpr int MI_X_DIM       = 52;   // Play x_batch (C1-C4)
constexpr int MI_BID_X_DIM   = 3;    // Bid x_batch

class ModelInference {
public:
    ModelInference() = default;

    bool loadModel(const std::string& name, const std::string& path) {
        try {
            torch::jit::Module module = torch::jit::load(path);
            module.eval();

            bool isFp16 = false;
            for (const auto& param : module.parameters()) {
                if (param.scalar_type() == at::kHalf) isFp16 = true;
                break;
            }
            if (isFp16) {
                module.to(torch::kFloat32);
            }

            models_[name] = std::move(module);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    void unloadModel(const std::string& name) {
        models_.erase(name);
    }

    bool isLoaded(const std::string& name) const {
        return models_.find(name) != models_.end();
    }

    // ── 叫分模型推理 ─────────────────────────────────────────────────
    // z: (B, 2, 120) flat — Row0=action broadcast, Row1=hand
    // x: (B, 3) flat — bidInfo 三人叫分原始值 (绝对位置, 由 feature_encoder.h buildBidFeatures 组装)
    // 分 chunk≤16 批量推理, 返回 argmax Q 值对应的动作索引
    int predictBid(const std::string& modelName,
                   const std::vector<float>& z,
                   const std::vector<float>& x,
                   int batchSize) {
        if (batchSize <= 0) return 0;
        auto it = models_.find(modelName);
        if (it == models_.end()) return 0;

        static constexpr int kChunk = 16;
        static constexpr int kZStride = MI_Z_ROWS * MI_Z_FEAT;  // 240
        static constexpr int kXStride = MI_BID_X_DIM;            // 3

        try {
            std::vector<torch::Tensor> outputs;
            int offset = 0;

            while (offset < batchSize) {
                int chunk = std::min(kChunk, batchSize - offset);

                auto z_t = torch::from_blob(
                    (void*)(z.data() + offset * kZStride),
                    {chunk, MI_Z_ROWS, MI_Z_FEAT}, torch::kFloat32).clone();
                auto x_t = torch::from_blob(
                    (void*)(x.data() + offset * kXStride),
                    {chunk, MI_BID_X_DIM}, torch::kFloat32).clone();

                torch::NoGradGuard no_grad;
                auto out = it->second.forward({z_t, x_t}).toTensor();
                outputs.push_back(out);
                offset += chunk;
            }

            auto all = torch::cat(outputs, 0);
            return all.argmax(0).item<int>();
        } catch (const std::exception&) {
            return 0;
        }
    }

    // ── 出牌模型推理 (V1 DouMH 兼容格式, main.cpp 当前使用) ──────────
    // z_batch: (B, 41, 120) flat — 共享 Transformer 编码 state+action+history
    // x_batch: (B, X_DIM) flat — 标量辅助特征 (52 或 56 维)
    // 分 chunk≤50 批量推理, 返回 argmax Q 值对应的合法动作索引
    int predictPlay(const std::string& modelName,
                    const std::vector<float>& z_batch,
                    const std::vector<float>& x_batch,
                    int batchSize) {
        if (batchSize <= 0) return 0;
        auto it = models_.find(modelName);
        if (it == models_.end()) return 0;

        // Infer x_dim from data size
        int xDim = (batchSize > 0 && !x_batch.empty())
            ? (int)x_batch.size() / batchSize : MI_X_DIM;

        static constexpr int kChunk = 50;
        static constexpr int kZStride = MI_ZB_ROWS * MI_Z_FEAT;  // 4920

        try {
            std::vector<torch::Tensor> outputs;
            int offset = 0;

            while (offset < batchSize) {
                int chunk = std::min(kChunk, batchSize - offset);

                auto z_t = torch::from_blob(
                    (void*)(z_batch.data() + offset * kZStride),
                    {chunk, MI_ZB_ROWS, MI_Z_FEAT}, torch::kFloat32).clone();
                auto x_t = torch::from_blob(
                    (void*)(x_batch.data() + offset * xDim),
                    {chunk, xDim}, torch::kFloat32).clone();

                torch::NoGradGuard no_grad;
                auto out = it->second.forward({z_t, x_t}).toTensor();
                outputs.push_back(out);
                offset += chunk;
            }

            auto all = torch::cat(outputs, 0);
            return all.argmax(0).item<int>();
        } catch (const std::exception&) {
            return 0;
        }
    }

    // ── 出牌模型推理 (V2 Cross-Attention 格式, 预留接口) ────────────
    // 与 V1 格式的关键区别: state/action/history 分离编码
    //   - state_z:   (1, 5, 120)  — 5 静态行 (自手牌/其他联合/已出×3), 不随动作变化
    //   - action_z:  (B, 1, 120)  — B 个候选动作各自编码
    //   - history_z: (1, 35, 120) — 35 手历史, 不随动作变化
    // Cross-Attention: action_z (Q) 交叉关注 state_z + history_z (K/V)
    // 当前 main.cpp 未使用 (使用 V1 predictPlay), 保留供未来升级
    int predictPlayCrossAttention(const std::string& modelName,
                                  const std::vector<float>& state_z,
                                  const std::vector<float>& action_z,
                                  const std::vector<float>& history_z,
                                  const std::vector<float>& x_batch,
                                  int batchSize) {
        if (batchSize <= 0) return 0;
        auto it = models_.find(modelName);
        if (it == models_.end()) return 0;

        int xDim = (batchSize > 0 && !x_batch.empty())
            ? (int)x_batch.size() / batchSize : MI_X_DIM;

        try {
            auto sz = torch::from_blob((void*)state_z.data(),
                {1, 5, 120}, torch::kFloat32).clone();
            auto hz = torch::from_blob((void*)history_z.data(),
                {1, 35, 120}, torch::kFloat32).clone();
            auto az = torch::from_blob((void*)action_z.data(),
                {batchSize, 1, 120}, torch::kFloat32).clone();
            auto xb = torch::from_blob((void*)x_batch.data(),
                {batchSize, xDim}, torch::kFloat32).clone();

            torch::NoGradGuard no_grad;
            auto out = it->second.forward({sz, az, hz, xb}).toTensor();
            return out.argmax(0).item<int>();
        } catch (const std::exception&) {
            return 0;
        }
    }

    // ── 角色 → 模型名映射 ───────────────────────────────────────────
    // Botzone 叫分顺序: bidPosition 0=first, 1=second, 2=third
    // 出牌角色: 自己与地主的相对关系 → landlord / landlord_down / landlord_up

    static const char* bidModelName(int bidPosition) {
        static const char* names[] = {"first", "second", "third"};
        return names[bidPosition % 3];
    }

    static const char* playModelName(int myPosition, int landlordPosition) {
        if (myPosition == landlordPosition) return "landlord";       // 自己是地主
        if (myPosition == (landlordPosition + 1) % 3) return "landlord_down"; // 地主下手
        return "landlord_up";                                        // 地主的对手(地主上手)
    }

    // 预加载模型: 首回合未确定地主时默认加载 landlord_down 覆盖大多数情况
    static const char* predictPlayModel(int /*bidPosition*/) {
        return "landlord_down";
    }

    // 模型文件路径: {dir}/{name}_jit.pt 或 {dir}/{name}_jit_fp16.pt
    static std::string modelPath(const std::string& dir,
                                  const std::string& name,
                                  bool fp16 = false) {
        std::string base;
        if (dir.empty()) {
            base = name;
        } else {
            char last = dir.back();
            if (last == '/' || last == '\\')
                base = dir + name;
            else
                base = dir + "/" + name;
        }
        return fp16 ? (base + "_jit_fp16.pt") : (base + "_jit.pt");
    }

private:
    std::unordered_map<std::string, torch::jit::Module> models_;
};

#endif // MODEL_INFERENCE_H
