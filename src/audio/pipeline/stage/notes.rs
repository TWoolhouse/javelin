use itertools::Itertools;

use crate::{
    audio::pipeline::Sample,
    audio::pipeline::module::{PassBuilder, PassSpec, PipelineModule, PostStage, PostStageBuilder},
};

#[derive(Debug, Clone, PartialEq)]
pub struct Notes {
    pub step: f32,
    pub low_freq: f32,
}

pub type StageNotes = Notes;

impl PostStage for StageNotes {
    fn process(&mut self, pass: PassBuilder) -> PassBuilder {
        // FIXME: Should use frequency not bin index.
        let mut bucket = self.low_freq.ceil();

        let samples = pass
            .samples
            .into_iter()
            .enumerate()
            .chunk_by(|(i, _)| {
                let b = bucket;
                if (*i as f32) < bucket {
                    b
                } else {
                    while (*i as f32) >= bucket {
                        bucket = (bucket * self.step).ceil();
                    }
                    bucket
                }
            })
            .into_iter()
            .map(|(_, chunk)| {
                chunk
                    .map(|(_, s)| s)
                    .reduce(|acc, s| Sample {
                        freq: s.freq.max(acc.freq),
                        db: s.db.max(acc.db),
                    })
                    .unwrap()
            })
            .collect();

        PassBuilder {
            samples,
            idx_max: None,
            ..pass
        }
    }
}

impl PipelineModule<PassSpec, PassSpec> for StageNotes {
    fn spec(&self, spec: PassSpec) -> PassSpec {
        // FIXME: Calculate the max_samples as this is an overestimate.
        spec
    }
}

impl PostStageBuilder for Notes {
    type Stage = StageNotes;

    fn build(&mut self, _spec: &PassSpec) -> Result<Self::Stage, ()> {
        Ok(self.clone())
    }
}
