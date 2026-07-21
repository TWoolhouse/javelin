use crate::{
    audio::pipeline::Pass,
    audio::{fft::FFTResolution, pipeline::Sample},
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PassSpec {
    pub samples: usize,
}
#[derive(Debug)]
pub struct PassBuilder {
    pub samples: Vec<Sample>,
    /// Index of the peak sample in the [samples](Self::samples) vector
    pub idx_max: Option<usize>,
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

impl PassBuilder {
    fn compute_max(&mut self) -> usize {
        let idx = self
            .samples
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.db.partial_cmp(&b.db).unwrap_or_else(|| {
                    if a.db.is_finite() {
                        std::cmp::Ordering::Greater
                    } else if b.db.is_finite() {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.idx_max = Some(idx);
        idx
    }

    fn max(&mut self) -> usize {
        self.idx_max.unwrap_or_else(|| self.compute_max())
    }
}

impl From<PassBuilder> for Pass {
    fn from(mut value: PassBuilder) -> Self {
        Self {
            peak: value.max(),
            samples: value.samples,
        }
    }
}
