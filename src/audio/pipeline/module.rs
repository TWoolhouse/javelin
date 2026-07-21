use crate::audio::{
    fft::FFTResolution,
    pipeline::pass::{PassBuilder, PassSpec},
};

/// Pipeline Module that consumes `UpstreamSpec` and produces `DownstreamSpec`
pub trait PipelineModule<UpstreamSpec, DownstreamSpec> {
    fn spec(&self, upstream: UpstreamSpec) -> DownstreamSpec;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CaptureSpec {
    pub resolution: FFTResolution,
}
#[derive(Debug)]
pub struct Capture<'s> {
    pub buffer: &'s mut [f32],
}

pub trait Capturer: PipelineModule<(), CaptureSpec> {
    fn capture(&mut self) -> Capture<'_>;
}

pub trait Transformer: PipelineModule<CaptureSpec, PassSpec> {
    fn process(&mut self, capture: Capture) -> PassBuilder;
}

pub trait PostStage: PipelineModule<PassSpec, PassSpec> {
    fn process(&mut self, pass: PassBuilder) -> PassBuilder;
}

pub trait PostStageBuilder {
    type Stage: PostStage;
    fn build(&mut self, spec: &PassSpec) -> Result<Self::Stage, ()>;
}
