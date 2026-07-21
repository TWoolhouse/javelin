use crate::audio::{
    fft::executor::{FFTExecutor, HANN_WINDOW_GAIN, hann_window},
    pipeline::{
        module::{Capture, CaptureSpec, PipelineModule, Transformer},
        pass::{PassBuilder, PassSpec, RawSamples},
    },
};

#[derive(Debug, Clone, PartialEq)]
pub struct FFTransformConfig {
    pub min_db: f32,
}

#[derive(Debug)]
pub struct FFTransform {
    hann_window: Vec<f32>,
    executor: FFTExecutor,
    power_factor: f32,
    min_power: f32,
    bin_width_hz: f32,
    samples_out: usize,
}

impl FFTransform {
    pub fn new(config: &FFTransformConfig, spec: &CaptureSpec) -> Self {
        let hann = hann_window(spec.resolution.samples());
        Self {
            hann_window: (0..spec.resolution.samples()).map(hann).collect(),
            executor: FFTExecutor::new(spec.resolution.samples()),
            power_factor: {
                let sample_count = spec.resolution.samples();
                2.0 / (sample_count * sample_count) as f32 / HANN_WINDOW_GAIN
            },
            min_power: 10f32.powf(config.min_db as f32 / 10.0),
            bin_width_hz: spec.resolution.hertz(),
            samples_out: spec.resolution.samples_out(),
        }
    }
}

impl Transformer for FFTransform {
    fn process(&mut self, capture: Capture) -> PassBuilder {
        let samples = (0..capture.buffer.len()).map(|i| capture.buffer[i] * self.hann_window[i]);

        let fft_samples = self.executor.process_samples(samples);

        let samples = fft_samples[..self.samples_out]
            .into_iter()
            .map(|c| {
                let power = c.norm_sqr() * self.power_factor;
                power.max(self.min_power).log10() * 10.0
            })
            .collect();

        PassBuilder {
            samples: RawSamples::from_decibels(samples),
            idx_max: None,
        }
    }
}

impl PipelineModule<CaptureSpec, PassSpec> for FFTransform {
    fn spec(&self, _upstream: CaptureSpec) -> PassSpec {
        PassSpec {
            frequencies: (0..self.samples_out)
                .map(|bin| bin as f32 * self.bin_width_hz)
                .collect(),
        }
    }
}
