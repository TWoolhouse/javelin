use crate::audio::pipeline::{
    capture::{DeviceCapture, DeviceCaptureConfig},
    module::{Capturer, PipelineModule, PostStage, Transformer},
    pass::Pass,
    stage::PostStage as PostStageConfig,
    transform::{FFTransform, FFTransformConfig},
};

pub mod capture;
pub mod module;
pub mod pass;
mod reconfigurable;
pub mod stage;
pub mod transform;
pub use reconfigurable::PipelineReconfigurable;

pub struct Pipeline {
    capturer: DeviceCapture,
    // pre_stages: Vec<Box<dyn PreStage>>, // TODO: Add pre-stages?
    transformer: FFTransform,
    post_stages: Vec<Box<dyn PostStage>>,
    frequencies: Vec<f32>,
}

impl Pipeline {
    pub fn new<'s>(
        capture: &mut DeviceCaptureConfig,
        transform: &mut FFTransformConfig,
        stages: impl IntoIterator<Item = &'s mut PostStageConfig>,
    ) -> Result<Self, ()> {
        let capturer = DeviceCapture::new(capture)?;
        let spec = capturer.spec(());
        let transformer = FFTransform::new(transform, &spec);
        let mut spec = transformer.spec(spec);

        let stages = stages.into_iter();
        let mut modules = Vec::with_capacity(stages.size_hint().0);
        for stage_config in stages {
            let stage = stage_config.build(&spec)?;
            spec = stage.spec(spec);
            modules.push(stage);
        }

        Ok(Self {
            capturer,
            // pre_stages: Vec::new(),
            transformer,
            post_stages: modules,
            frequencies: spec.frequencies,
        })
    }

    pub fn run(&mut self) -> Pass {
        let capture = self.capturer.capture();
        let mut pass = self.transformer.process(capture);
        for stage in self.post_stages.iter_mut() {
            pass = stage.process(pass);
        }
        pass.build(&self.frequencies)
    }
}
