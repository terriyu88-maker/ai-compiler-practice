# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.runtime.runtime_utils import assert_tensor_metadata

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_0 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(6L); x1+=static_cast<int64_t>(1L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(784L); x2+=static_cast<int64_t>(16L))
                    {
                        {
                            if(C10_LIKELY(x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(784L)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x2 + 784L*x1 + 4704L*x0), static_cast<int64_t>(16));
                                auto tmp1 = in_ptr0[static_cast<int64_t>(x1)];
                                auto tmp4 = in_ptr1[static_cast<int64_t>(x1)];
                                auto tmp10 = in_ptr2[static_cast<int64_t>(x1)];
                                auto tmp13 = in_ptr3[static_cast<int64_t>(x1)];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 - tmp2;
                                auto tmp5 = static_cast<float>(1e-05);
                                auto tmp6 = float(tmp4 + tmp5);
                                auto tmp7 = 1 / std::sqrt(tmp6);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp3 * tmp8;
                                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                                auto tmp12 = tmp9 * tmp11;
                                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                auto tmp15 = tmp12 + tmp14;
                                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                                tmp16.store(in_out_ptr0 + static_cast<int64_t>(x2 + 784L*x1 + 4704L*x0));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(6L); x1+=static_cast<int64_t>(1L))
                {
                    #pragma GCC ivdep
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(14L); x2+=static_cast<int64_t>(1L))
                    {
                        #pragma GCC ivdep
                        for(int64_t x3=static_cast<int64_t>(0L); x3<static_cast<int64_t>(14L); x3+=static_cast<int64_t>(1L))
                        {
                            {
                                {
                                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(2L*x3 + 56L*x2 + 784L*x1 + 4704L*x0)];
                                    auto tmp1 = in_out_ptr0[static_cast<int64_t>(1L + 2L*x3 + 56L*x2 + 784L*x1 + 4704L*x0)];
                                    auto tmp3 = in_out_ptr0[static_cast<int64_t>(28L + 2L*x3 + 56L*x2 + 784L*x1 + 4704L*x0)];
                                    auto tmp5 = in_out_ptr0[static_cast<int64_t>(29L + 2L*x3 + 56L*x2 + 784L*x1 + 4704L*x0)];
                                    auto tmp2 = max_propagate_nan(tmp0, tmp1);
                                    auto tmp4 = max_propagate_nan(tmp2, tmp3);
                                    auto tmp6 = max_propagate_nan(tmp4, tmp5);
                                    out_ptr0[static_cast<int64_t>(x1 + 6L*x3 + 84L*x2 + 1176L*x0)] = tmp6;
                                }
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(16L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(6L); x1+=static_cast<int64_t>(16L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(25L); x2+=static_cast<int64_t>(16L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0L) && x1 < static_cast<int64_t>(6L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(16L)))
                            {
                                alignas(std::max(std::size_t(16), alignof(float))) float tmp1[16*16];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(6L); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x2 + 25L*x1 + 25L*x1_inner + 150L*x0), static_cast<int64_t>(16));
                                    tmp0.store(tmp1 + static_cast<int64_t>(16L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(6L),static_cast<int64_t>(16),false>(tmp1, static_cast<int64_t>(16), out_ptr1 + static_cast<int64_t>(x1 + 6L*x2 + 150L*x0), static_cast<int64_t>(6L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0L) && x1 < static_cast<int64_t>(6L) && x2 >= static_cast<int64_t>(16L) && x2 < static_cast<int64_t>(25L)))
                            {
                                alignas(std::max(std::size_t(16), alignof(float))) float tmp1[16*16];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(6L); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x2 + 25L*x1 + 25L*x1_inner + 150L*x0), static_cast<int64_t>(9L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(9L*x1_inner), static_cast<int64_t>(9L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(6L),static_cast<int64_t>(9L),false>(tmp1, static_cast<int64_t>(9L), out_ptr1 + static_cast<int64_t>(x1 + 6L*x2 + 150L*x0), static_cast<int64_t>(6L));
                            }
                        }
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(6400L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(16L); x1+=static_cast<int64_t>(16L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(16L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 16L*x0), static_cast<int64_t>(16));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp2 * tmp7;
                            auto tmp10 = tmp8 * tmp9;
                            auto tmp12 = tmp10 + tmp11;
                            auto tmp13 = at::vec::clamp_min(tmp12, decltype(tmp12)(0));
                            tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 16L*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(5L); x1+=static_cast<int64_t>(1L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(5L); x2+=static_cast<int64_t>(16L))
                    {
                        for(int64_t x3=static_cast<int64_t>(0L); x3<static_cast<int64_t>(16L); x3+=static_cast<int64_t>(16L))
                        {
                            {
                                if(C10_LIKELY(x2 >= static_cast<int64_t>(0L) && x2 < static_cast<int64_t>(5L) && x3 >= static_cast<int64_t>(0) && x3 < static_cast<int64_t>(16L)))
                                {
                                    alignas(std::max(std::size_t(16), alignof(float))) float tmp7[16*16];
                                    for (long x2_inner = 0; x2_inner < static_cast<int64_t>(5L); x2_inner++)
                                    {
                                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x3 + 32L*x2 + 32L*x2_inner + 320L*x1 + 1600L*x0), static_cast<int64_t>(16));
                                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(16L + x3 + 32L*x2 + 32L*x2_inner + 320L*x1 + 1600L*x0), static_cast<int64_t>(16));
                                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(160L + x3 + 32L*x2 + 32L*x2_inner + 320L*x1 + 1600L*x0), static_cast<int64_t>(16));
                                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(176L + x3 + 32L*x2 + 32L*x2_inner + 320L*x1 + 1600L*x0), static_cast<int64_t>(16));
                                        auto tmp2 = at::vec::maximum(tmp0, tmp1);
                                        auto tmp4 = at::vec::maximum(tmp2, tmp3);
                                        auto tmp6 = at::vec::maximum(tmp4, tmp5);
                                        tmp6.store(tmp7 + static_cast<int64_t>(16L*x2_inner));
                                    }
                                    transpose_mxn<float,static_cast<int64_t>(5L),static_cast<int64_t>(16),false>(tmp7, static_cast<int64_t>(16), out_ptr0 + static_cast<int64_t>(x2 + 5L*x1 + 25L*x3 + 400L*x0), static_cast<int64_t>(25L));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused_relu_2 = async_compile.cpp_pybinding(['float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(7680L); x0+=static_cast<int64_t>(16L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(7680L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp1.store(in_out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused_relu_3 = async_compile.cpp_pybinding(['float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(5376L); x0+=static_cast<int64_t>(16L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(5376L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp1.store(in_out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1 = args
        args.clear()
        assert_size_stride_grouped((arg2_1, arg0_1, arg1_1), ((64, 1, 28, 28), (6, 1, 5, 5), (6, )), ((784, 784, 28, 1), (25, 25, 5, 1), (1, )), 'input')
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:1
        buf0 = extern_kernels.convolution(arg2_1, arg0_1, arg1_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
        assert_tensor_metadata(buf0, (64, 6, 28, 28), (4704, 784, 28, 1), torch.float32, 'torch.ops.aten.convolution.default')
        del arg0_1
        del arg1_1
        del arg2_1
        assert_size_stride_grouped((arg3_1, arg4_1, arg5_1, arg6_1), ((6, ), (6, ), (6, ), (6, )), ((1, ), (1, ), (1, ), (1, )), 'input')
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cpu((64, 6, 14, 14), (1176, 1, 84, 6), torch.float32)
        assert_size_stride(arg7_1, (16, 6, 5, 5), (150, 25, 5, 1), 'input')
        buf3 = empty_strided_cpu((16, 6, 5, 5), (150, 1, 30, 6), torch.float32)
        assert_size_stride(arg8_1, (16, ), (1, ), 'input')
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_0:2
        cpp_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_0(buf1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, buf2, buf3)
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del buf1
        # Topologically Sorted Source Nodes: [batch_norm, x, x_1, conv2d_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices, aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:3
        buf4 = extern_kernels.convolution(buf2, buf3, arg8_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
        assert_tensor_metadata(buf4, (64, 16, 10, 10), (1600, 1, 160, 16), torch.float32, 'torch.ops.aten.convolution.default')
        del arg8_1
        del buf2
        del buf3
        assert_size_stride_grouped((arg9_1, arg10_1, arg11_1, arg12_1), ((16, ), (16, ), (16, ), (16, )), ((1, ), (1, ), (1, ), (1, )), 'input')
        buf5 = buf4; del buf4  # reuse
        buf6 = empty_strided_cpu((64, 16, 5, 5), (400, 25, 5, 1), torch.float32)
        assert_size_stride_grouped((arg14_1, arg13_1), ((120, ), (120, 400)), ((1, ), (400, 1)), 'input')
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1:4
        cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1(buf5, arg9_1, arg10_1, arg11_1, arg12_1, buf6)
        del arg10_1
        del arg11_1
        del arg12_1
        del arg9_1
        del buf5
        buf7 = empty_strided_cpu((64, 120), (120, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_1, x_2, x_3, x_4, linear], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices, aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:5
        extern_kernels.addmm(arg14_1, reinterpret_tensor(buf6, (64, 400), (400, 1), 0), reinterpret_tensor(arg13_1, (400, 120), (1, 400), 0), alpha=1, beta=1, out=buf7)
        del arg13_1
        del arg14_1
        del buf6
        buf8 = buf7; del buf7  # reuse
        assert_size_stride_grouped((arg16_1, arg15_1), ((84, ), (84, 120)), ((1, ), (120, 1)), 'input')
        # [Provenance debug handles] cpp_fused_relu_2:6
        cpp_fused_relu_2(buf8)
        buf9 = empty_strided_cpu((64, 84), (84, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, linear_1], Original ATen: [aten.relu, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:7
        extern_kernels.addmm(arg16_1, buf8, reinterpret_tensor(arg15_1, (120, 84), (1, 120), 0), alpha=1, beta=1, out=buf9)
        del arg15_1
        del arg16_1
        del buf8
        buf10 = buf9; del buf9  # reuse
        assert_size_stride_grouped((arg18_1, arg17_1), ((10, ), (10, 84)), ((1, ), (84, 1)), 'input')
        # [Provenance debug handles] cpp_fused_relu_3:8
        cpp_fused_relu_3(buf10)
        buf11 = empty_strided_cpu((64, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6, linear_2], Original ATen: [aten.relu, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:9
        extern_kernels.addmm(arg18_1, buf10, reinterpret_tensor(arg17_1, (84, 10), (1, 84), 0), alpha=1, beta=1, out=buf11)
        del arg17_1
        del arg18_1
        return (buf11, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((6, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((6, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, 1, 28, 28), (784, 784, 28, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((6, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((6, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((6, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((6, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((16, 6, 5, 5), (150, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((120, 400), (400, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((84, 120), (120, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((84, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((10, 84), (84, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((10, ), (1, ), device='cpu', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cpu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
