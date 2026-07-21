use crate::{
    audio::pipeline::module::{Capture, CaptureSpec, Capturer, PipelineModule},
    audio::{
        fft::{FFTBuffer, FFTResolution},
        stream::{Device, Stream},
    },
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCaptureConfig {
    pub device: Device,
    pub resolution: FFTResolution,
}

#[derive(Debug)]
pub struct DeviceCapture {
    resolution: FFTResolution,
    stream: Stream,
}

impl DeviceCapture {
    pub fn new(config: &mut DeviceCaptureConfig) -> Self {
        config.resolution = config
            .resolution
            .with_sample_rate(config.device.config.sample_rate as usize)
            .optimal();
        let (fft_tx, fft_rx) = FFTBuffer::new(config.resolution.samples());
        Self {
            resolution: config.resolution,
            stream: config
                .device
                .create_stream_as_input_from_output_device(fft_tx, fft_rx)
                .unwrap(), // FIXME: This can fail in normal operation
        }
    }
}

impl Capturer for DeviceCapture {
    fn capture(&mut self) -> Capture<'_> {
        Capture {
            buffer: self.stream.buffer.slice(),
        }
    }
}

impl PipelineModule<(), CaptureSpec> for DeviceCapture {
    fn spec(&self, _upstream: ()) -> CaptureSpec {
        CaptureSpec {
            resolution: self.resolution,
        }
    }
}
