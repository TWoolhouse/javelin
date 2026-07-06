use crate::audio::{
    fft::{FFTBuffer, processor::FFTProcessor},
    pipeline::config::ConfigBuilder,
    stream::Stream,
};

pub mod config;
use config::Config;
mod pass;
pub use pass::FFTPass;
use rodio::math::db_to_linear;

#[derive(Debug)]
pub struct Pipeline {
    config: Config,
    processor: FFTProcessor,
    stream: Stream,
}

#[derive(Debug, Clone)]
pub struct Sample {
    /// Frequency in Hz
    pub freq: f32,
    /// Decibels
    pub db: f32,
    /// Linear amplitude, see [db_to_linear]
    pub amp: f32,
}

impl Sample {
    /// [db_to_linear]
    pub fn amp(&self) -> f32 {
        db_to_linear(self.db)
    }
}

impl Pipeline {
    pub fn new(config: impl Into<ConfigBuilder>) -> Self {
        let config = config.into().build();

        let (fft_tx, fft_rx) = FFTBuffer::new(config.resolution.samples());

        // FIXME: propergate errors
        Self {
            processor: FFTProcessor::new(config.resolution.samples()),
            stream: config
                .device
                .create_stream_as_input_from_output_device(fft_tx, fft_rx)
                .unwrap(),
            config,
        }
    }

    pub fn reconfigure(&mut self, config: impl Into<ConfigBuilder>) {
        let config = config.into().build();

        macro_rules! delta {
            ($old:expr, $new:expr, $($attr:tt)+) => {
                $old.$($attr)+ != $new.$($attr)+
            };
        }

        if delta!(self.config, config, resolution.samples()) {
            self.processor.resize(config.resolution.samples());
        }
        if delta!(self.config, config, resolution.samples()) || delta!(self.config, config, device)
        {
            let (fft_tx, fft_rx) = FFTBuffer::new(config.resolution.samples());
            self.stream = config
                .device
                .create_stream_as_input_from_output_device(fft_tx, fft_rx)
                .unwrap();
        }

        self.config = config;
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
}

// Other impl's
mod fft;
// mod history;
