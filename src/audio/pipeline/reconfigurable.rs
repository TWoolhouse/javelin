use std::time::Duration;

use derive_more::{Deref, DerefMut};

use crate::{
    audio::pipeline::{
        Pipeline,
        capture::DeviceCaptureConfig,
        stage::{PostStage as PostStageConfig, notes::Notes, smooth::Smooth},
        transform::FFTransformConfig,
    },
    audio::{fft::FFTResolution, stream::Device},
};

#[derive(Deref, DerefMut)]
pub struct PipelineReconfigurable {
    #[deref]
    #[deref_mut]
    pipeline: Pipeline,

    pub capturer: DeviceCaptureConfig,
    pub transform: FFTransformConfig,
    pub stages: Vec<PostStageConfig>,
}

impl PipelineReconfigurable {
    pub fn new(
        mut capturer: DeviceCaptureConfig,
        mut transform: FFTransformConfig,
        mut stages: Vec<PostStageConfig>,
    ) -> Self {
        // TODO: Cloning the configs feels wrong
        Self {
            pipeline: Pipeline::new(&mut capturer, &mut transform, stages.iter_mut()),
            stages,
            capturer,
            transform,
        }
    }

    pub fn reconfigure(&mut self) {
        // TODO: We don't have to throw away the entire pipeline, we can reconfigure it in place.
        // But for now, just rebuild it.
        self.pipeline = Pipeline::new(
            &mut self.capturer,
            &mut self.transform,
            self.stages.iter_mut(),
        );
    }
}

impl Default for PipelineReconfigurable {
    fn default() -> Self {
        Self::new(
            DeviceCaptureConfig {
                device: Device::try_default().unwrap(),
                resolution: FFTResolution::from_duration(Duration::from_millis(40), 44_100),
            },
            FFTransformConfig { min_db: -200.0 },
            vec![
                Notes {
                    step: 1.06,
                    low_freq: 1.0,
                }
                .into(),
                Smooth { bias_new: 0.5 }.into(),
            ],
        )
    }
}
