use crate::audio::pipeline::{
    capture::{DeviceCapture, DeviceCaptureConfig},
    module::{Capturer, PipelineModule, PostStage, Transformer},
    stage::PostStage as PostStageConfig,
    transform::{FFTransform, FFTransformConfig},
};

pub mod capture;
pub mod module;
mod reconfigurable;
pub mod stage;
pub mod transform;
pub use reconfigurable::PipelineReconfigurable;
use rodio::math::{db_to_linear, linear_to_db};

#[derive(Debug, Clone)]
pub struct Sample {
    /// Frequency in Hz
    pub freq: f32,
    /// Decibels
    pub db: f32,
}

impl Sample {
    /// [db_to_linear]
    pub fn amp(&self) -> f32 {
        db_to_linear(self.db)
    }

    pub fn with_amp(&self, amp: f32) -> Self {
        Self {
            freq: self.freq,
            db: linear_to_db(amp),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Pass {
    pub samples: Vec<Sample>,
    /// Index of the peak sample in the [samples](Self::samples) vector
    pub peak: usize,
}

pub struct Pipeline {
    capturer: DeviceCapture,
    // pre_stages: Vec<Box<dyn PreStage>>, // TODO: Add pre-stages?
    transformer: FFTransform,
    post_stages: Vec<Box<dyn PostStage>>,
}

impl Pipeline {
    pub fn new<'s>(
        capture: &mut DeviceCaptureConfig,
        transform: &mut FFTransformConfig,
        stages: impl IntoIterator<Item = &'s mut PostStageConfig>,
    ) -> Self {
        let capturer = DeviceCapture::new(capture);
        let spec = capturer.spec(());
        let transformer = FFTransform::new(transform, &spec);
        let mut spec = transformer.spec(spec);

        let stages = stages.into_iter();
        let mut modules = Vec::with_capacity(stages.size_hint().0);
        for stage_config in stages {
            let stage = stage_config.build(&spec).unwrap(); // TODO: Handle errors
            spec = stage.spec(spec);
            modules.push(stage);
        }

        Self {
            capturer,
            // pre_stages: Vec::new(),
            transformer,
            post_stages: modules,
        }
    }

    pub fn run(&mut self) -> Pass {
        let capture = self.capturer.capture();
        let mut pass = self.transformer.process(capture);
        for stage in self.post_stages.iter_mut() {
            pass = stage.process(pass);
        }
        pass.into()
    }
}

impl Pass {
    pub fn peak_sample(&self) -> &Sample {
        &self.samples[self.peak]
    }
}
