use std::time::{Duration, Instant};

use float_ord::FloatOrd;

use crate::audio::pipeline::{
    module::{PipelineModule, PostStage, PostStageBuilder},
    pass::{PassBuilder, PassSpec, Sample},
};

#[derive(Debug, Clone, PartialEq)]
pub struct History {
    pub cull_duration: Duration,
}

#[derive(Debug)]
pub struct StageHistory {
    config: History,
    frequencies: Vec<f32>,
    buckets: Vec<(Instant, Sample)>,
}

impl PostStage for StageHistory {
    fn process(&mut self, mut pass: PassBuilder) -> PassBuilder {
        let peak_idx = pass.max_idx().unwrap_or_default();
        let peak_val = pass.samples.as_inner()[peak_idx];

        let peak_sample = if pass.samples.is_decibels() {
            Sample {
                freq: self.frequencies[peak_idx],
                db: peak_val,
            }
        } else {
            Sample::from_amp(self.frequencies[peak_idx], peak_val)
        };

        let now = Instant::now();

        // Cull old buckets and add the new peak sample to the history
        let instant_cull = now - self.config.cull_duration;
        let pp = self
            .buckets
            .partition_point(|(instant, _)| *instant < instant_cull);
        self.buckets.drain(..pp);
        self.buckets.push((now, peak_sample.clone()));

        // Find the maximum peak sample in the history
        let max_peak = self
            .buckets
            .iter()
            .map(|(_, s)| s)
            .max_by(|a, b| FloatOrd(a.db).cmp(&FloatOrd(b.db)))
            .unwrap_or(&peak_sample);

        pass.peak = Some(max_peak.clone());
        pass
    }
}

impl PipelineModule<PassSpec, PassSpec> for StageHistory {
    fn spec(&self, input: PassSpec) -> PassSpec {
        input
    }
}

impl PostStageBuilder for History {
    type Stage = StageHistory;

    fn build(&mut self, spec: &PassSpec) -> Result<Self::Stage, ()> {
        Ok(Self::Stage {
            config: self.clone(),
            frequencies: spec.frequencies.clone(),
            buckets: Vec::new(),
        })
    }
}
