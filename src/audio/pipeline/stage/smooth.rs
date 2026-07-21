use crate::audio::pipeline::{
    module::{PipelineModule, PostStage, PostStageBuilder},
    pass::{PassBuilder, PassSpec},
};

#[derive(Debug, Clone, PartialEq)]
pub struct Smooth {
    pub bias_new: f32,
}

#[derive(Debug, Clone)]
pub struct StageSmooth {
    config: Smooth,
    buffer: Vec<f32>,
}

impl PostStage for StageSmooth {
    fn process(&mut self, mut pass: PassBuilder) -> PassBuilder {
        for (sample, smooth) in pass
            .samples
            .as_amplitudes()
            .iter_mut()
            .zip(self.buffer.iter_mut())
        {
            *smooth += (*sample - *smooth) * (self.config.bias_new);
            *sample = *smooth;
        }

        pass
    }
}

impl PipelineModule<PassSpec, PassSpec> for StageSmooth {
    fn spec(&self, input: PassSpec) -> PassSpec {
        input
    }
}

impl PostStageBuilder for Smooth {
    type Stage = StageSmooth;
    fn build(&mut self, spec: &PassSpec) -> Result<Self::Stage, ()> {
        self.bias_new = self.bias_new.clamp(0.0, 1.0);
        Ok(Self::Stage {
            config: self.clone(),
            buffer: vec![0.0; spec.sample_count()],
        })
    }
}
