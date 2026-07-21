use std::ops::RangeInclusive;

use itertools::Itertools;

use crate::audio::pipeline::{
    module::{PipelineModule, PostStage, PostStageBuilder},
    pass::{PassBuilder, PassSpec},
};

#[derive(Debug, Clone, PartialEq)]
pub struct Notes {
    pub step: f32,
    pub low_freq: f32,
}

#[derive(Debug)]
pub struct StageNotes {
    mapping: Vec<(RangeInclusive<usize>, f32)>,
}

impl PostStage for StageNotes {
    fn process(&mut self, mut pass: PassBuilder) -> PassBuilder {
        let samples = pass.samples.as_inner_mut();

        for (i, (range, _)) in self.mapping.iter().enumerate() {
            let sample = samples[range.clone()]
                .iter()
                .max_by(|&a, &b| {
                    a.partial_cmp(b).unwrap_or_else(|| {
                        if a.is_finite() {
                            std::cmp::Ordering::Greater
                        } else if b.is_finite() {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    })
                })
                .unwrap();
            samples[i] = *sample;
        }
        samples.truncate(self.mapping.len());

        PassBuilder {
            samples: pass.samples,
            ..pass
        }
    }
}

impl PipelineModule<PassSpec, PassSpec> for StageNotes {
    fn spec(&self, _spec: PassSpec) -> PassSpec {
        PassSpec {
            frequencies: self.mapping.iter().map(|i| i.1).collect(),
        }
    }
}

impl PostStageBuilder for Notes {
    type Stage = StageNotes;

    fn build(&mut self, spec: &PassSpec) -> Result<Self::Stage, ()> {
        let mut bucket = self.low_freq.ceil();

        let mapping = spec
            .frequencies
            .iter()
            .enumerate()
            .chunk_by(|(_, f)| {
                while **f >= bucket {
                    bucket = (bucket * self.step).ceil();
                }
                bucket
            })
            .into_iter()
            .map(|(f, mut it)| {
                let range = {
                    let (first, _) = it.next().unwrap();
                    let (last, _) = it.last().unwrap_or((first, &0.0));
                    first..=last
                };
                (range, f)
            })
            .collect();

        Ok(StageNotes { mapping })
    }
}
