use clap::Command;
use cudarc::driver::{LaunchAsync, LaunchConfig};

pub const TRAIN_SUBCOMMAND: &str = "train";
pub fn cli() -> Command {
    Command::new(TRAIN_SUBCOMMAND)
        .about("Trains the model (currently only on a single NVidia GPU) using the build tokenizer and downloaded data files.")
}

pub fn train() -> Result<(), Box<dyn std::error::Error>> {
    // TEMPORARY: CUDA testing
    let device = cudarc::driver::CudaDevice::new(0)?;

    let input = device.htod_copy(vec![1.5f32; 100])?;
    let mut out = device.alloc_zeros::<f32>(100)?;

    let ptx = cudarc::nvrtc::compile_ptx("
extern \"C\" __global__ void sin_kernel(float *out, const float *inp, const size_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}")?;

    device.load_ptx(ptx, "my_module", &["sin_kernel"])?;

    let sin_kernel = device.get_func("my_module", "sin_kernel").unwrap();
    let config = LaunchConfig::for_num_elems(100);
    unsafe { sin_kernel.launch(config, (&mut out, &input, 100usize)) }?;

    let out_host: Vec<f32> = device.dtoh_sync_copy(&out)?;
    assert_eq!(out_host, [1.5; 100].map(f32::sin));

    Ok(())
}