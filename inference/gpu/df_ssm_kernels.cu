/**
 * DF-SSM v9 — cuBLAS INT8 scaffold + custom conv/SSM
 *
 * Scaffold matmul: torch._int_mm (cuBLAS INT8 tensor cores, 624 TOPS)
 * Custom kernels only for: conv step, SSM step (stateful ops)
 * Everything else (norm, gate, LoRA, quantize): PyTorch ops
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define D_INNER  4096
#define D_XBC    4352
#define D_STATE  128
#define NHEADS   64
#define HEADDIM  32
#define NGROUPS  1
#define D_CONV   4

// ===================================================================
// Conv1d step — (batch, d_xBC) state update
// ===================================================================

__global__ void conv_step_kernel(
    half* __restrict__ conv_state,    // (batch, d_xBC, 3)
    half* __restrict__ x_new,         // (batch, d_xBC)
    const half* __restrict__ conv_weight, // (d_xBC, 4)
    const half* __restrict__ conv_bias,   // (d_xBC,)
    int d_xBC, int x_stride
) {
    int bi = blockIdx.y;
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= d_xBC) return;
    half* cs = conv_state + bi * d_xBC * 3;
    half* xn = x_new + bi * x_stride;  // x_new may be strided (within proj)
    float s0=__half2float(cs[ch*3]),s1=__half2float(cs[ch*3+1]),s2=__half2float(cs[ch*3+2]);
    float xv=__half2float(xn[ch]);
    cs[ch*3]=__float2half(s1); cs[ch*3+1]=__float2half(s2); cs[ch*3+2]=__float2half(xv);
    float w0=__half2float(conv_weight[ch*D_CONV]),w1=__half2float(conv_weight[ch*D_CONV+1]);
    float w2=__half2float(conv_weight[ch*D_CONV+2]),w3=__half2float(conv_weight[ch*D_CONV+3]);
    float out=s0*w0+s1*w1+s2*w2+xv*w3+__half2float(conv_bias[ch]);
    float sig=1.0f/(1.0f+expf(-out));
    xn[ch]=__float2half(out*sig);
}

// ===================================================================
// SSM step — parallel over d_state
// ===================================================================

__global__ void ssm_step_kernel(
    half* __restrict__ h_state,         // (batch, NHEADS, HEADDIM, D_STATE)
    const half* __restrict__ x_ssm,     // (batch, x_stride) — SSM input within proj
    const half* __restrict__ B_param,   // (batch, B_stride) — B within proj
    const half* __restrict__ C_param,   // (batch, C_stride) — C within proj
    const half* __restrict__ dt_raw,    // (batch, dt_stride) — dt within proj
    const float* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    half* __restrict__ y_out,           // (batch, D_INNER)
    int x_stride, int B_offset, int C_offset, int dt_stride, int dt_offset
) {
    int bi = blockIdx.y;
    int hp_idx = blockIdx.x;
    int head = hp_idx / HEADDIM;
    int n = threadIdx.x;
    if (head >= NHEADS || n >= D_STATE) return;

    // Pointers into contiguous (batch, proj_dim) layout
    const half* my_x = x_ssm + bi * x_stride;
    const half* my_B = B_param + bi * x_stride + B_offset;
    const half* my_C = C_param + bi * x_stride + C_offset;
    const half* my_dt = dt_raw + bi * dt_stride + dt_offset;

    float dt_v = __half2float(my_dt[head]) + dt_bias[head];
    dt_v = log1pf(expf(dt_v));
    float A_bar = expf(dt_v * (-expf(A_log[head])));
    float xval = __half2float(my_x[hp_idx]);
    float X = xval * dt_v;
    float b = __half2float(my_B[n]);
    float c = __half2float(my_C[n]);

    int h_off = bi*NHEADS*HEADDIM*D_STATE + hp_idx*D_STATE + n;
    float h = __half2float(h_state[h_off]);
    h = A_bar*h + X*b;
    h_state[h_off] = __float2half(h);

    float partial = c * h;
    for (int off=16;off>0;off>>=1)
        partial += __shfl_down_sync(0xffffffff, partial, off);

    extern __shared__ char sm[];
    float* ws = (float*)sm;
    int wid=n/32, lid=n%32;
    if(lid==0) ws[wid]=partial;
    __syncthreads();
    if(n==0) {
        float y = ws[0]+ws[1]+ws[2]+ws[3] + xval*D_param[head];
        y_out[bi*D_INNER+hp_idx] = __float2half(y);
    }
}

// ===================================================================
// Thin wrappers for Python
// ===================================================================

void conv_step(
    torch::Tensor conv_state,  // (batch, d_xBC, 3)
    torch::Tensor xBC,         // (batch, d_xBC) — contiguous slice
    torch::Tensor conv_weight, // (d_xBC, 4)
    torch::Tensor conv_bias,   // (d_xBC,)
    int d_xBC
) {
    int batch = conv_state.size(0);
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    dim3 grid((d_xBC+255)/256, batch);
    conv_step_kernel<<<grid, 256, 0, s>>>(
        reinterpret_cast<half*>(conv_state.data_ptr<at::Half>()),
        reinterpret_cast<half*>(xBC.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(conv_weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(conv_bias.data_ptr<at::Half>()),
        d_xBC, xBC.stride(0));
}

void ssm_step(
    torch::Tensor h_state,     // (batch, NHEADS, HEADDIM, D_STATE)
    torch::Tensor xBC,         // (batch, d_xBC) contiguous after conv
    torch::Tensor dt,          // (batch, nheads) — contiguous slice
    torch::Tensor dt_bias,
    torch::Tensor A_log,
    torch::Tensor D_param,
    torch::Tensor y_out        // (batch, D_INNER)
) {
    int batch = h_state.size(0);
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    dim3 grid(NHEADS*HEADDIM, batch);

    ssm_step_kernel<<<grid, D_STATE, 4*sizeof(float), s>>>(
        reinterpret_cast<half*>(h_state.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xBC.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xBC.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xBC.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(dt.data_ptr<at::Half>()),
        dt_bias.data_ptr<float>(), A_log.data_ptr<float>(),
        D_param.data_ptr<float>(),
        reinterpret_cast<half*>(y_out.data_ptr<at::Half>()),
        xBC.stride(0), D_INNER, D_INNER + NGROUPS*D_STATE,
        dt.stride(0), 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DF-SSM v9 — conv + SSM kernels (scaffold via cuBLAS INT8)";
    m.def("conv_step", &conv_step);
    m.def("ssm_step", &ssm_step);
}
