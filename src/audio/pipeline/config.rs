use std::time::Duration;

use crate::audio::{fft::FFTResolution, stream::Device};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub db_min: i32,
    pub device: Device,
    pub resolution: FFTResolution,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigBuilder {
    base: Config,
    db_min: Option<i32>,
    device: Option<Device>,
    resolution: Option<FFTResolution>,
}

impl From<ConfigBuilder> for Config {
    fn from(builder: ConfigBuilder) -> Self {
        let device = builder.device.unwrap_or(builder.base.device);
        let db_min = builder.db_min.unwrap_or(builder.base.db_min);
        let sample_rate = device.config.sample_rate as usize;
        let resolution = builder
            .resolution
            .unwrap_or(builder.base.resolution)
            .with_sample_rate(sample_rate)
            .optimal();

        assert!(
            db_min <= 0,
            "db_min={} must be less than or equal to 0",
            db_min
        );
        assert_eq!(
            resolution.sample_rate(),
            sample_rate,
            "resolution.sample_rate={} must match that of the device={}",
            resolution.sample_rate(),
            sample_rate
        );
        assert_ne!(
            resolution.samples(),
            0,
            "resolution.samples={} must be greater than 0",
            resolution.samples()
        );

        Self {
            db_min,
            device,
            resolution,
        }
    }
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Config::default_internal().into()
    }

    pub fn db_min(mut self, db_min: i32) -> Self {
        self.db_min = Some(db_min);
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn resolution(mut self, resolution: FFTResolution) -> Self {
        self.resolution = Some(resolution);
        self
    }

    pub fn build(self) -> Config {
        self.into()
    }
}

impl Config {
    fn default_internal() -> Self {
        Self {
            db_min: -200,
            device: Device::try_default().expect("Failed to get default device"),
            resolution: FFTResolution::from_duration(Duration::from_millis(40), 44_100),
        }
    }

    pub fn builder(self) -> ConfigBuilder {
        self.into()
    }
}

impl From<Config> for ConfigBuilder {
    fn from(config: Config) -> Self {
        Self {
            base: config,
            db_min: None,
            device: None,
            resolution: None,
        }
    }
}
