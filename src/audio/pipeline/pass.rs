use crate::audio::pipeline::Sample;

#[derive(Debug, Clone)]
pub struct FFTPass {
    pub samples: Vec<Sample>,
    /// Index of the peak sample in the [samples](Self::samples) vector
    pub peak: usize,
}

impl FFTPass {
    pub fn peak_sample(&self) -> &Sample {
        &self.samples[self.peak]
    }
}
